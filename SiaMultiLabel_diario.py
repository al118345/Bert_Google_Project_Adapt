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



    try:
        for prediction in predictions:
            resultado.append((in_sentences[indice], prediction['probabilities'], labels[prediction['labels']]))

            print(str(indice) +"###"+str(len(in_sentences)))
            indice = indice + 1
            '''
            if indice>100:
                return resultado
            '''

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

    #print(estimator.evaluate(input_fn=test_input_fn, steps=None))




    '''
    pred_sentences = conexionBase.Obtenertweetisentimiento(911000)
    auxiliar=[]
    for i in pred_sentences:
            auxiliar.append(i[0])

    sentimiento = []
    for z in pred_sentences:
            sentimiento.append(z[1])
    predictions = getPrediction(auxiliar)

    df = pd.DataFrame(list(predictions), sentimiento)

    writer = pd.ExcelWriter("test_polaridad" + '.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name="Analisis")
    workbook = writer.book
    writer.save()
    '''

    '''
    auxiliar = []


    sentimiento = []

    indice=0
    for i in todos:
        if indice < (len(todos) - 350):
            pass
        else:
            if (i[8] == 'positivo'):
                auxiliar.append(i[0])
                sentimiento.append(1)
            if (i[8] == 'negativo'):
                auxiliar.append(i[0])
                sentimiento.append(0)
            if (i[8] == 'neutral'):
                auxiliar.append(i[0])
                sentimiento.append(2)
        indice = indice + 1
    predictions = getPrediction(auxiliar)
    df = pd.DataFrame(list(predictions), sentimiento)

    writer = pd.ExcelWriter("parte_test_polaridad" + '.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name="Analisis")
    workbook = writer.book
    writer.save()

    auxiliar = []

    sentimiento = []

    indice = 0
    for i in todos:
        if (i[8] == 'positivo'):
                auxiliar.append(i[0])
                sentimiento.append(1)
        if (i[8] == 'negativo'):
                auxiliar.append(i[0])
                sentimiento.append(0)
        if (i[8] == 'neutral'):
                auxiliar.append(i[0])
                sentimiento.append(2)

    predictions = getPrediction(auxiliar)
    df = pd.DataFrame(list(predictions), sentimiento)

    writer = pd.ExcelWriter("parte_entrenamiento" + '.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name="Analisis")
    workbook = writer.book
    writer.save()
    '''
    '''
    calculate auc multiclass python
    parte de fechas
    '''

    conexionBase = ConexionBaseDeDatos()
    anterior=datetime(2019, 4,1, 1, 30)
    result = {}
    result["positivo"] = []
    result["neutral"] = []
    result["negativo"] = []
    result["total"] = []
    result["precio"] = []
    result["sentimiento_positivo"] = []
    result["sentimiento_negativo"] = []
    result["fecha"] = []

    indice=0
    for timestamp in datespan( datetime(2019, 4,1, 2, 30),  datetime(2019, 7,31, 23, 30),delta=timedelta(hours=1)):

        print("Iteracion"+str(timestamp))
        if conexionBase.existe_hora(timestamp):
            pass
        else:
            pred_sentences = conexionBase.ObtenerEntreHoras(anterior, timestamp)
            auxiliar = []
            for i in pred_sentences:
                auxiliar.append(i[0])
            sentimiento = []
            positivo=0
            negativo=0

            for z in pred_sentences:
                if z[1]=='positive':
                    positivo=positivo+1
                else:
                    negativo=negativo+1
                sentimiento.append(z[1])

            if len(auxiliar)>0:
                predictions = getPrediction(auxiliar)
                '''print(type(predictions).__name__)'''
                df = pd.DataFrame(list(predictions), sentimiento[0:len(predictions)])
                precio = conexionBase.ObtenerPrecio(timestamp)
                result["positivo"].append(len(df.loc[df[2] == 1]))
                result["neutral"].append(len(df.loc[df[2] == 2]))
                result["negativo"].append(len(df.loc[df[2] == 0]))
                result["total"].append(len(pred_sentences))
                for total_precio in precio:
                    result["precio"].append(total_precio[0].replace(".", ","))
                    break
                result["sentimiento_positivo"].append(positivo)
                result["sentimiento_negativo"].append(negativo)
                result["fecha"].append(str(timestamp))
                conexionBase.almacenar_datos_valoracion(result["precio"][indice],result["positivo"][indice],result["negativo"][indice],  result["neutral"][indice], result["sentimiento_negativo"] [indice], result["sentimiento_positivo"][indice],  result["total"][indice],result["fecha"][indice])
            else:
                precio = conexionBase.ObtenerPrecio(timestamp)
                result["positivo"].append(0)
                result["neutral"].append(0)
                result["negativo"].append(0)
                result["total"].append(0)
                for total_precio in precio:
                    result["precio"].append(total_precio[0].replace(".", ","))
                    break
                result["sentimiento_positivo"].append(positivo)
                result["sentimiento_negativo"].append(negativo)
                result["fecha"].append(str(timestamp))
                conexionBase.almacenar_datos_valoracion(result["precio"][indice],result["positivo"][indice],result["negativo"][indice],  result["neutral"][indice], result["sentimiento_negativo"] [indice], result["sentimiento_positivo"][indice],  result["total"][indice],result["fecha"][indice])
            indice=indice+1
        anterior = timestamp
