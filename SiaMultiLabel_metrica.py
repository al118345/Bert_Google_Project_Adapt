from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from tensorflow import keras
import os
import re
from ConexionBaseDeDatos import ConexionBaseDeDatos
from ExtractData import ExtractData
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from datetime import date, datetime, timedelta


'''
Rate cambiado
'''
def datespan(startDate, endDate, delta=timedelta(days=1)):
    currentDate = startDate
    while currentDate < endDate:
        yield currentDate
        currentDate += delta


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

'''

create_model hace esto a continuación. 
Primero, vuelve a cargar el módulo concentrador BERT tf (esta vez para extraer el gráfico 
de cálculo). A continuación, crea una nueva capa única que será entrenada para adaptar BERT 
a nuestra tarea de sentimiento (es decir, clasificar si una crítica de película es positiva 
o negativa). Esta estrategia de usar un modelo mayormente entrenado se llama ajuste fino.
'''
def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  """Creates a classification model."""

  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, rate=1-0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)


'''
A continuación, incluiremos nuestra función de modelo en una función model_fn_builder
 que adapta nuestro modelo para que funcione con fines de capacitación, evaluación y predicción.
'''
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            (loss, predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                '''
                f1_score = tf.contrib.metrics.f1_score(
                    label_ids,
                    predicted_labels)
                '''
                '''
                auc = tf.metrics.auc(
                    label_ids,
                    predicted_labels)
                '''
                recall = tf.metrics.recall(
                    label_ids,
                    predicted_labels)
                precision = tf.metrics.precision(
                    label_ids,
                    predicted_labels)
                true_pos = tf.metrics.true_positives(
                    label_ids,
                    predicted_labels)
                true_neg = tf.metrics.true_negatives(
                    label_ids,
                    predicted_labels)
                false_pos = tf.metrics.false_positives(
                    label_ids,
                    predicted_labels)
                false_neg = tf.metrics.false_negatives(
                    label_ids,
                    predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    #"f1_score": f1_score,
                    #"auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


def getPrediction(in_sentences):
    labels = [0, 1, 2]
    input_examples = [run_classifier.InputExample(guid="", text_a=x, text_b=None, label=0) for x in
                      in_sentences]  # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH,
                                                       is_training=False, drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)
    indice=0
    resultado = []



    #
    try:
        for prediction in predictions:
            #elimian


            #

            from sklearn.preprocessing import label_binarize
            resultado.append((in_sentences[indice], prediction['probabilities'], label_binarize([labels[prediction['labels']]], classes=[0, 1, 2])[0] ))
            print(str(indice) +"###"+str(len(in_sentences)))
            indice = indice + 1


    except Exception as e:
        print(e)
        return resultado

    return resultado
    '''
    try:
        return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in
                zip(in_sentences, predictions)]
    except Exception as e:
        print(e)
    '''





if __name__ == '__main__':
    '''
    Configuración
    '''
    OUTPUT_DIR = 'Modelo_Sia'#@param {type:"string"}
    # Compute train and warmup steps from batch size
    # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3.0
    # Warmup is a period of time where hte learning rate
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100

    # Compute train and warmup steps from batch size
    # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3.0
    # Warmup is a period of time where hte learning rate
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100

    # Compute # train and warmup steps from batch size



    '''
    Genero el test y el train, importante todos los twwets  son lower
    '''
    DATA_COLUMN = 'sentence'
    LABEL_COLUMN = 'polarity'
    label_list = [0, 1, 2]
    datos= ExtractData()
    tweets=datos.cargarDatos()
    '''
    Fin carga de datos
     '''


    '''
    
    Data Preprocessing
    Necesitaremos transformar nuestros datos en un formato que BERT entienda. 
    Esto implica dos pasos. 
    
    Primero, creamos InputExample usando el constructor provisto en la biblioteca BERT.
        text_a es el texto que queremos clasificar, 
        etiqueta es la etiqueta de nuestro ejemplo, es decir, Verdadero, Falso
    
    '''
    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train_InputExamples = tweets[0].apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                 # Globally unique ID for bookkeeping, unused in this example
                                                                                 text_a=x[DATA_COLUMN],
                                                                                 text_b=None,
                                                                                 label=x[LABEL_COLUMN]), axis=1)

    test_InputExamples = tweets[1].apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                               text_a=x[DATA_COLUMN],
                                                                               text_b=None,
                                                                               label=x[LABEL_COLUMN]), axis=1)

    '''
       A continuación, debemos preprocesar nuestros datos para que coincidan con los datos 
       en los que BERT recibió capacitación. Para esto, tendremos que hacer un par de cosas :
       
            Minúsculas en nuestro texto (si estamos usando un modelo de minúsculas BERT)
            Tokenize (es decir, "sally dice hola" -> ["sally", "dice", "hola"))
            Divida las palabras en WordPieces (es decir, "llamando" -> ["call", "## ing"])
            Asigne nuestras palabras a los índices utilizando un archivo de vocabulario que proporciona BERT
            Agregue tokens especiales "CLS" y "SEP" (consulte el archivo Léame)
            Agregue tokens de "índice" y "segmento" a cada entrada (vea el documento BERT)
            Afortunadamente, no tenemos que preocuparnos por la mayoría de estos detalles.
    
        Para comenzar, necesitaremos cargar un archivo de vocabulario y 
        una información de minúsculas directamente desde el módulo del concentrador BERT tf:
    '''
    BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    tokenizer = create_tokenizer_from_hub_module()
    print(tokenizer.tokenize("This here's an example of using the BERT tokenizer"))

    '''
    Usando nuestro tokenizer, llamaremos run_classifier.convert_examples_to_features
     en nuestros InputExamples para convertirlos en características que BERT entiende.
    '''

    # We'll set sequences to be at most 128 tokens long.
    ''' Modifico a 300 '''
    MAX_SEQ_LENGTH = 300
    # Convert our train and test features to InputFeatures that BERT understands.
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH,
                                                                      tokenizer)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH,
                                                                     tokenizer)

    '''
    importante verificar
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    '''
    num_train_steps = int(len(train_features))
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
    model_fn = model_fn_builder(
        num_labels=len(label_list),
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})

    '''A continuación, creamos una función de creación de entradas que toma nuestro
    conjunto de funciones de entrenamiento (train_features) y produce un generador. 
    Este es un patrón de diseño bastante estándar para trabajar con Tensorflow Estimators.'''

    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)
    print("Beginning Training!")
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)
    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)



    conexionBase = ConexionBaseDeDatos()
    todos = conexionBase.conectar_al_traintest()
    auxiliar = []

    sentimiento = []
    indice=0
    for i in todos:
        #comprobar con todos
        if indice > (len(todos) - 350):
            if (i[8] == 'positivo'):
                auxiliar.append(i[0])
                sentimiento.append(1)
            if (i[8] == 'negativo'):
                auxiliar.append(i[0])
                sentimiento.append(0)
            if (i[8] == 'neutral'):
                auxiliar.append(i[0])
                sentimiento.append(2)
        indice=indice+1

    predictions = getPrediction(auxiliar)
    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc

    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    from sklearn.preprocessing import label_binarize
    sentimiento = label_binarize(sentimiento[:len(predictions)], classes=[0, 1, 2])

    aux=[]
    for i in predictions:
        aux.append(i[2])

    from numpy import array

    from sklearn.metrics import(brier_score_loss, precision_score, recall_score, f1_score)
    for i in range(len(label_list)):
        fpr[i], tpr[i], _ = roc_curve(sentimiento[:, i], array(aux)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print("\tPrecision: %1.3f" % precision_score(sentimiento[:, i], array(aux)[:, i]))
        print("\tRecall: %1.3f" % recall_score(sentimiento[:, i], array(aux)[:, i]))
        print("\tF1: %1.3f\n" % f1_score(sentimiento[:, i], array(aux)[:, i]))
    # Compute micro-average ROC curve and ROC area


    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(label_list))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(label_list)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(label_list)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(label_list)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)


    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(label_list)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(label_list)):
        precision[i], recall[i], _ = precision_recall_curve(sentimiento[:, i], array(aux)[:, i])
        average_precision[i] = average_precision_score(sentimiento[:, i], array(aux)[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(sentimiento[:, i], array(aux)[:, i])
    average_precision["micro"] = average_precision_score(sentimiento[:, i], array(aux)[:, i],
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    from inspect import signature
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                     **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))

    from itertools import cycle

    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(len(label_list)), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.show()
