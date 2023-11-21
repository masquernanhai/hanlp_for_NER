# hanlp_for_NER
1、修改模型加载路径：site-packages\hanlp\utils\io_util.py, 在hanlp_home_default的return处改为模型路径。或者在linux上export HANLP_HOME = 文件路径
2、自定义词典：在模型文件夹中找到config.json文件夹后修改其中的dictionary，将自定义词典以K:V键值的形式写入
3、调用模型使用ner.py
4、调用模型训练使用ner_fine_tunning.py
5、配置文件可以指定多任务或者单任务，同时指定加载的相应模型
