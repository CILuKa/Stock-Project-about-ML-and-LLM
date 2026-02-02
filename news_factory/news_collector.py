"""
this script is a tool to collect news about stock and the relate policies about stock market and nation.
主要难点：
1.高效可实用的搜索引擎api

2.规范搜索格式

3.巨大规模搜索结果的处理及存储（按照日期的）

4.合适的网站或许可以解决一些搜索压力

注意事项：
1.规范输出

2.每日的或每一段时间的自动更新
"""

from Collect_and_Storage.Collect import RSSCollector
from Collect_and_Storage.Storage import NewsStorage