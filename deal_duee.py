import json


def trans(filelist, output):
    lst = []
    for filename in filelist:
        with open(filename, encoding="utf-8") as f:
            for line in f:
                file_obj = json.loads(line)
                obj = {"id": file_obj["id"], "text": file_obj["text"], "cls_label": "", "spo_list": []}
                # obj["cls_label"] = ?
                order = 0
                try:
                    for item in file_obj["event_list"]:
                        sub_obj = item['arguments'][0]
                        obj_obj = item['arguments'][1]
                        try:
                            s_start = file_obj['text'].index(sub_obj['argument'])
                            s_end = int(file_obj['text'].index(sub_obj['argument'])) + len(sub_obj['argument'])
                        except:
                            print(file_obj['text'] + "\tNOT INCLUDE\t" + sub_obj['argument'])
                        try:
                            o_start = file_obj['text'].index(obj_obj['argument'])
                            o_end = int(file_obj['text'].index(obj_obj['argument'])) + len(obj_obj['argument'])
                        except:
                            print(file_obj['text'] + "\tNOT INCLUDE\t" + obj_obj['argument'])
                        spo_obj = {
                            "order": order,
                            "s": {
                                "text": sub_obj['argument'],
                                "start": s_start,
                                "end": s_end,
                                "type": sub_obj['role']
                            },
                            "p": item['trigger'],  # 原文中表示关系的词
                            # "p1": item['class'] if item['class'] else "",  # 关系所属分类
                            "o": {
                                "text": obj_obj['argument'],
                                "start": o_start,
                                "end": o_end,
                                "type": obj_obj['role']
                            }
                        }
                        obj["spo_list"].append(spo_obj)
                        order += 1
                except:  # 没有事件
                    pass
                lst.append(obj)
            # f.close()
    # print(len(lst))
    with open(output, "a+", encoding="utf-8") as des:
        des.writelines(str(lst))
        des.close()


if __name__ == "__main__":
    file_list = ["./duee_fin_dev.json", "./duee_fin_train.json"]
    trans(file_list, "./our_fin.json")
