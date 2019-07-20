import json
JsonFile = open('/home/superuser/SequenceEncoding/Representations/multiling_english.json')
values = json.load(JsonFile)

documents = {}
sentences = []

sentences_num = len(values['data']['train'] )
for sentence in values['data']['train']:
    text = sentence['text']
    document_id = sentence['document_index']
    label = sentence['labels']
    #print text,document_id
    if document_id not in documents.keys():
        documents[document_id] = []
    documents[document_id].append((text,label))


f1 = open("/home/superuser/SequenceEncoding/Representations/dataset.txt", "w+")
f2 = open("/home/superuser/SequenceEncoding/Representations/sentences.txt", "w+")
for document in documents.keys():
    sentences = documents[document]
    f1.write('Document '+ str(document) + ' ' + str(len(sentences)))
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