from summa.summarizer import summarize
from summa import keywords


text = "Neo-Nazism consists of post-World War II militant social or political movements seeking to revive and implement the ideology of Nazism. Neo-Nazis seek to employ their ideology to promote hatred and attack minorities, or in some cases to create a fascist political state. It is a global phenomenon, with organized representation in many countries and international networks. It borrows elements from Nazi doctrine, including ultranationalism, racism, xenophobia, ableism, homophobia, anti-Romanyism, antisemitism, anti-communism and initiating the Fourth Reich. Holocaust denial is a common feature, as is the incorporation of Nazi symbols and admiration of Adolf Hitler. In some European and Latin American countries, laws prohibit the expression of pro-Nazi, racist, anti-Semitic, or homophobic views. Many Nazi-related symbols are banned in European countries (especially Germany) in an effort to curtail neo-Nazism. The term neo-Nazism describes any post-World War II militant, social or political movements seeking to revive the ideology of Nazism in whole or in part. The term neo-Nazism can also refer to the ideology of these movements, which may borrow elements from Nazi doctrine, including ultranationalism, anti-communism, racism, ableism, xenophobia, homophobia, anti-Romanyism, antisemitism, up to initiating the Fourth Reich. Holocaust denial is a common feature, as is the incorporation of Nazi symbols and admiration of Adolf Hitler. Neo-Nazism is considered a particular form of far-right politics and right-wing extremism."

print(summarize(text, ratio=0.2))

summarize(text, words=50)

print("Keywords:\n"
,keywords.keywords(text))
# to print the top 3 keywords
print("Top 3 Keywords:\n",keywords.keywords(text,words=3))