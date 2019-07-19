import json
JsonFile = open('/home/superuser/SequenceEncoding/Representations/dailymail-small.json')
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


f = open("/home/superuser/SequenceEncoding/Representations/dataset.txt", "a")

for document in documents.keys():
    sentences = documents[document]
    f.write('Document '+ str(document) + ' ' + str(len(sentences)))
    f.write('\n')
    for sentence in sentences:
        sentence_text = sentence[0]
        sentence_text = sentence_text.encode('utf8')
        sentence_label = sentence[1]
        f.write("[%s]" % sentence_text)
        f.write("[%s]\n" % sentence_label)
f.close()
JsonFile.close()