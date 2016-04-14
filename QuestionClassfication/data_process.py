try:
	import xml.etree.cElementTree as ET
except ImportError:
	import xml.etree.ElementTree as ET
import data_helpers

def write_to_file(file,line):
	file.write(line+"\n")

def cat_map():
	catmap={}
	id=1
	f=open("cat")
	cat=set([s.strip() for s in list(f.readlines())])
	for i in cat:
		catmap[i]=id
		id=id+1
	return catmap

tree = ET.ElementTree(file="test.xml")
root = tree.getroot()
cnn=open("cnn","a")
lstm=open("lstm","a")
cat=open("cat","a")
for vespaadd in root:
	document = vespaadd.find("document")
	if(document!=None):
		subject = document.find("subject")
		content = document.find("content")
		maincat = document.find("maincat")
		if(subject==None):
			continue
		if(content==None):
			content=subject
		if(maincat==None):
			continue
		write_to_file(cnn,data_helpers.clean_str(subject.text))
		write_to_file(lstm,data_helpers.clean_str(content.text))
		write_to_file(cat,data_helpers.clean_str(maincat.text))
cnn.close()
lstm.close()
cat.close()