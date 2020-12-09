#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import csv
import json
import os
import sys

from multiprocessing import cpu_count

import absl.app
import progressbar
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from six.moves import zip

from .util.config import Config, initialize_globals
from .util.checkpoints import load_graph_for_evaluation
from .util.evaluate_tools import calculate_and_print_report, save_samples_json
from .util.feeding import create_dataset
from .util.flags import create_flags, FLAGS
from .util.helpers import check_ctcdecoder_version
from .util.logging import create_progressbar, log_error, log_progress

check_ctcdecoder_version()


def sparse_tensor_value_to_texts(value, alphabet):
    r"""
    Given a :class:`tf.SparseTensor` ``value``, return an array of Python strings
    representing its values, converting tokens to strings using ``alphabet``.
    """
    return sparse_tuple_to_texts((value.indices, value.values, value.dense_shape), alphabet)


def sparse_tuple_to_texts(sp_tuple, alphabet):
    indices = sp_tuple[0]
    values = sp_tuple[1]
    results = [[] for _ in range(sp_tuple[2][0])]
    for i, index in enumerate(indices):
        results[index[0]].append(values[i])
    # List of strings
    return [alphabet.Decode(res) for res in results]


def get_tower_results(iterator):
    r'''
    With this preliminary step out of the way, we can for each GPU introduce a
    tower for which's batch we calculate and return the optimization gradients
    and the average loss across towers.
    '''
    # To calculate the mean of the losses
    logits = []

    with tfv1.variable_scope(tfv1.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(Config.available_devices)):
            # Execute operations of tower i on device i
            device = Config.available_devices[i]
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i):
                    # Calculate the avg_loss and mean_edit_distance and retrieve the decoded
                    # batch along with the original batch's labels (Y) of this tower
                    avg_loss, non_finite_files = calculate_mean_edit_distance_and_loss(iterator, dropout_rates,
                                                                                       reuse=i > 0)

                    # Allow for variables to be re-used by the next tower
                    tfv1.get_variable_scope().reuse_variables()

                    # Retain tower's avg losses
                    tower_avg_losses.append(avg_loss)

                    # Compute gradients for model parameters using tower's mini-batch
                    gradients = optimizer.compute_gradients(avg_loss)

                    # Retain tower's gradients
                    tower_gradients.append(gradients)

                    tower_non_finite_files.append(non_finite_files)

    avg_loss_across_towers = tf.reduce_mean(input_tensor=tower_avg_losses, axis=0)
    tfv1.summary.scalar(name='step_loss', tensor=avg_loss_across_towers, collections=['step_summaries'])

    all_non_finite_files = tf.concat(tower_non_finite_files, axis=0)

    # Return gradients and the average loss
    return tower_gradients, avg_loss_across_towers, all_non_finite_files


def evaluate(test_csvs, create_model, csv_file=None, csv_file_obj=None):
    if FLAGS.scorer_path:
        scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                        FLAGS.scorer_path, Config.alphabet)
    else:
        scorer = None

    test_sets = [create_dataset([csv],
                                batch_size=FLAGS.test_batch_size,
                                train_phase=False,
                                reverse=FLAGS.reverse_test,
                                limit=FLAGS.limit_test) for csv in test_csvs]
    iterator = tfv1.data.Iterator.from_structure(tfv1.data.get_output_types(test_sets[0]),
                                                 tfv1.data.get_output_shapes(test_sets[0]),
                                                 output_classes=tfv1.data.get_output_classes(test_sets[0]))
    test_init_ops = [iterator.make_initializer(test_set) for test_set in test_sets]

    tower_batch_wav_filename = []
    tower_transposed = []
    tower_batch_x_len = []
    tower_batch_y = []
    with tfv1.variable_scope(tfv1.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(Config.available_devices)):
            # Execute operations of tower i on device i
            device = Config.available_devices[i]
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i):
                    batch_wav_filename, (batch_x, batch_x_len), batch_y = iterator.get_next()
                    tower_batch_wav_filename.append(batch_wav_filename)
                    tower_batch_x_len.append(batch_x_len)
                    tower_batch_y.append(batch_y)

                    # One rate per layer
                    no_dropout = [None] * 6
                    logits, _ = create_model(batch_x=batch_x,
                                             seq_length=batch_x_len,
                                             dropout=no_dropout,
                                             reuse=i > 0)

                    # Transpose to batch major and apply softmax for decoder
                    transposed = tf.nn.softmax(tf.transpose(a=logits, perm=[1, 0, 2]))
                    tower_transposed.append(transposed)
                    tfv1.get_variable_scope().reuse_variables()

    tfv1.train.get_or_create_global_step()

    # Get number of accessible CPU cores for this process
    try:
        num_processes = cpu_count()
    except NotImplementedError:
        num_processes = 1

    with tfv1.Session(config=Config.session_config) as session:
        if FLAGS.model_path:
            with tfv1.gfile.FastGFile(FLAGS.model_path, 'rb') as fin:
                graph_def = tfv1.GraphDef()
                graph_def.ParseFromString(fin.read())

            var_names = [v.name for v in tfv1.trainable_variables()]
            var_tensors = tfv1.import_graph_def(graph_def, return_elements=var_names)

            # build a { var_name: var_tensor } dict
            var_tensors = dict(zip(var_names, var_tensors))
            training_graph = tfv1.get_default_graph()

            assign_ops = []
            for name, restored_tensor in var_tensors.items():
                training_tensor = training_graph.get_tensor_by_name(name)
                assign_ops.append(tfv1.assign(training_tensor, restored_tensor))

            init_from_frozen_model_op = tfv1.group(*assign_ops)
            session.run(init_from_frozen_model_op)
        else:
            load_graph_for_evaluation(session)

        def run_test(init_op, dataset):
            wav_filenames = []
            predictions = []
            ground_truths = []

            bar = create_progressbar(prefix='Test epoch | ',
                                     widgets=['Steps: ', progressbar.Counter(), ' | ', progressbar.Timer()]).start()
            log_progress('Test epoch...')

            step_count = 0

            # Initialize iterator to the appropriate dataset
            session.run(init_op)

            # First pass, compute losses and transposed logits for decoding
            while True:
                try:
                    batch_wav_filenames, batch_logits, batch_lengths, batch_transcripts = \
                        session.run([tower_batch_wav_filename, tower_transposed, tower_batch_x_len, tower_batch_y])
                except tf.errors.OutOfRangeError:
                    break

                for i in range(len(Config.available_devices)):
                    decoded = ctc_beam_search_decoder_batch(batch_logits[i], batch_lengths[i], Config.alphabet, FLAGS.beam_width,
                                                            num_processes=num_processes, scorer=scorer,
                                                            cutoff_prob=FLAGS.cutoff_prob, cutoff_top_n=FLAGS.cutoff_top_n)
                    predictions.extend(d[0][1] for d in decoded)
                    ground_truths.extend(sparse_tensor_value_to_texts(batch_transcripts[i], Config.alphabet))
                    wav_filenames.extend(
                        os.path.basename(wav_filename.decode('UTF-8')) for wav_filename in batch_wav_filenames[i])

                    step_count += 1
                    bar.update(step_count)
                    if csv_file:
                        csv_file.writerows(zip(wav_filenames, predictions))
                        wav_filenames = []
                        predictions = []
                if csv_file_obj:
                    csv_file_obj.flush()

            bar.finish()
            if not FLAGS.save_csv:
                # Print test summary
                return calculate_and_print_report(wav_filenames, ground_truths, predictions, dataset)
            return list(zip(wav_filenames, predictions))

        predictions = []
        for csv, init_op in zip(test_csvs, test_init_ops):
            print('Testing model on {}'.format(csv))
            result = run_test(init_op, dataset=csv)
            predictions.extend(result)
        return predictions


def main(_):
    initialize_globals()
    out_csv = None
    out_f = None
    if FLAGS.test_output_file and FLAGS.save_csv:
        out_f = open(FLAGS.test_output_file, 'w', encoding='utf-8')
        out_csv = csv.writer(out_f)
        out_csv.writerow(['wav_filename', 'prediction'])
    if not FLAGS.test_files:
        log_error('You need to specify what files to use for evaluation via '
                  'the --test_files flag.')
        sys.exit(1)

    from .train import create_model  # pylint: disable=cyclic-import,import-outside-toplevel
    samples = evaluate(FLAGS.test_files.split(','), create_model, out_csv, out_f)

    if out_f:
        out_f.close()
    elif FLAGS.test_output_file:
        save_samples_json(samples, FLAGS.test_output_file)
        print('saving json')


def run_script():
    create_flags()
    absl.app.run(main)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    run_script()
