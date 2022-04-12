# Python code to illustrate parsing of XML files
# importing the required modules
import xml.etree.ElementTree as et
import xmltodict
from xml.dom.minidom import parse
import xml.dom.minidom


def parse_xml(xmlfile):
    DOMTree = xml.dom.minidom.parse(xmlfile)
    collection = DOMTree.documentElement
    for item in collection.getElementsByTagName("problem"):
        description = item.getElementsByTagName('description')[0]
        print(description.childNodes[0].data)
        file = item.getElementsByTagName('file')[0]
        print("file:", file.childNodes[0].data)
        line = item.getElementsByTagName('line')[0]
        print("line:", line.childNodes[0].data)


def main():
    # parse xml file
    parse_xml('../../inspection/SyntaxError.xml')


if __name__ == "__main__":
    # calling main function
    main()
