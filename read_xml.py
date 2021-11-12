import xml.etree.ElementTree as ET
import csv


def read_and_write(xml_path, csv_path, aw='w'):
    tree = ET.parse(xml_path, parser=ET.XMLParser(encoding='iso-8859-5'))
    root = tree.getroot()

    result = dict()

    audio_number = 0
    max_audio = -1

    for data_set in root[1:]:
        for child in data_set:
            if child.tag == "data_set_id":
                audio_number += 1
                if audio_number == max_audio:
                    break
                if "path" not in result.keys():
                    result["path"] = []
            
                path = child.text[48:] # remove /home/... etc.
                if ',' in path:
                    continue
                result["path"].append(path)

                if "genre" not in result.keys():
                    result["genre"] = []
                result["genre"].append(path.split('/')[1])
            else:
                feature_name = ""
                values = []
                for subchild in child:
                    if subchild.tag == "name":
                        feature_name = subchild.text
                    else:
                        values.append(subchild.text.replace(',', '.'))
                if len(values) == 1:
                    if feature_name not in result.keys():
                        result[feature_name] = []
                    result[feature_name].append(values[0])
                elif len(values) > 1:
                    continue
                    for i in range(len(values)):
                        new_feature_name = feature_name + str(i)
                        if new_feature_name not in result.keys():
                            result[new_feature_name] = []
                        result[new_feature_name].append(values[i])


        if audio_number == max_audio:
                break

    with open(csv_path, aw) as f:
        key_array = list(result.keys())
        if(aw == 'w'):
            for k in range(len(key_array)):
                if k < len(key_array)-1:
                    f.write("%s,"%(key_array[k]))
                else:
                    f.write("%s"%(key_array[k]))
            f.write('\n')
        for k in range(len(result['path'])):
            for key_idx in range(len(key_array)):
                key = key_array[key_idx]
                if(k >= len(result[key])):
                    print("error at ", key, "k = ", k, "len = ",len(result[key]))
                if(key_idx < len(key_array)-1):
                    f.write("%s,"%(result[key][k]))
                else:
                    f.write("%s"%(result[key][k]))
            f.write('\n')

read_and_write("extracted_feature_values.xml", "extracted_features.csv", "w")
read_and_write("extracted_feature_values_2.xml", "extracted_features.csv", "a")
read_and_write("extracted_feature_values_3.xml", "extracted_features.csv", "a")
read_and_write("extracted_feature_values_4.xml", "extracted_features.csv", "a")
read_and_write("extracted_feature_values_5.xml", "extracted_features.csv", "a")
read_and_write("extracted_feature_values_6.xml", "extracted_features.csv", "a")
#read_and_write("extracted_feature_values_7.xml", "extracted_features.csv", "a")

