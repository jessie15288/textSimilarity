import json
import os
import csv
import tensorflow as tf
'''
fname = "/Users/jyang/datasets/nlp/text_similarity/oppo/oppp.json"

with open(fname, 'r', encoding="utf-8") as f:
    data = json.load(f)

    # data结构里面包含【train, test, dev】
    # 其中train（17万条）和dev（1万条）包含句子对和label，而test（5万条）只包含句子对。label里1表示句子是同义的，0表示非同义
    for i, sample in enumerate(data["dev"]):
        if i < 100:
            print(i, sample["q1"])
            print(i, sample["q2"])
            #print(i, sample["label"])
'''

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SimProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        train_data = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        for index, train in enumerate(train_data):
            guid = 'train-%d' % index
            text_a = train[0]
            text_b = train[1]
            label = train[2]
            print(text_a)
            print(text_b)
            print(label)


    def get_dev_examples(self, data_dir):
        '''
        file_path = os.path.join(data_dir, 'dev.csv')
        dev_df = pd.read_csv(file_path, encoding='utf-8')
        dev_data = []
        for index, dev in enumerate(dev_df.values):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(dev[0]))
            text_b = tokenization.convert_to_unicode(str(dev[1]))
            label = str(dev[2])
            dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return dev_data
        '''

    def get_test_examples(self, data_dir):
        '''
        file_path = os.path.join(data_dir, 'test.csv')
        test_df = pd.read_csv(file_path, encoding='utf-8')
        test_data = []
        for index, test in enumerate(test_df.values):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(test[0]))
            text_b = tokenization.convert_to_unicode(str(test[1]))
            label = str(test[2])
            test_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return test_data
        '''

    def get_sentence_examples(self, questions):
        '''
        for index, data in enumerate(questions):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(data[0]))
            text_b = tokenization.convert_to_unicode(str(data[1]))
            label = str(0)
            yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        '''

    def get_labels(self):
        '''
        return ['0', '1']
        '''

solu = SimProcessor()
data_dir = "/Users/jyang/datasets/nlp/text_similarity/lcqmc"
solu.get_train_examples(data_dir)