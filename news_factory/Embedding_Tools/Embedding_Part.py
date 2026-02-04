from sentence_transformers import SentenceTransformer as SBert


model = SBert('roberta-large-nli-stsb-mean-tokens')

default_address = ""
#按照日期读取文件，每日文件存储在一个json文件里

class Embedder:

    def __init__(self):
        self.News_address = default_address

    def File_2_Vec(self, News_address, filename):
        """
        读取文件内容为内存中数据进行处理
        Args:
            News_address:json文件存储地址
            filename:自动获取json地址下连续的json文件名

        Returns:
            Vectors:对应的向量(应当存储在搜索优化向量数据库中)

        """


