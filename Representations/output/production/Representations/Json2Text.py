import json
import sys


def main():
    JsonFile = open('/home/superuser/SequenceEncoding/Representations/multiling_english_lblratio2mod_oversample.json')
    values = json.load(JsonFile)

    documents = {}
    sentences = []

    mode = "train"
    for arg in sys.argv:
        if arg == "-test":
            mode = "test"
    print('Transforming '+str(mode)+'ing data from .json to .dat...')
    sentences_num = len(values['data'][mode])
    print('\t'+str(sentences_num)+' sentences')
    for sentence in values['data'][mode]:
        text = sentence['text']
        text = text.replace('\n', ' ') #remove line breaks inside sentences.
        '''
        text = text.replace('.',' ')
        text = text.replace('(',' ')
        text = text.replace(')',' ')
        text = text.replace('\t',' ')
        text = text.replace('{',' ')
        text = text.replace('}',' ')
        text = text.replace(',',' ')
        text = text.replace(':',' ')
        text = text.replace('?',' ')
        text = text.replace('!',' ')
        '''
        for ignored in xrange (0,10):
            text = text.replace('  ',' ')

        document_id = sentence['document_index']
        label = sentence['labels']
        #print text,document_id
        if document_id not in documents.keys():
            documents[document_id] = []
        documents[document_id].append((text,label))


    f1 = open("/home/superuser/SequenceEncoding/Representations/"+mode+"_dataset.txt", "w+")
    f2 = open("/home/superuser/SequenceEncoding/Representations/"+mode+"_sentences.txt", "w+")
    for document in documents.keys():
        sentences = documents[document]
        f1.write(mode+'/Document '+ str(document) + ' ' + str(len(sentences)))
        f1.write('\n')
        for sentence in sentences:
            sentence_text = sentence[0]
            sentence_text = sentence_text.encode('utf8')
            sentence_label = sentence[1]
            f1.write("[%s]" % sentence_text)
            f1.write("[%s]\n" % sentence_label)
            f2.write("%s " % sentence_text)
    f1.close()
    f2.close()
    JsonFile.close()
    print('...Done')
if __name__ == '__main__':
    main()
