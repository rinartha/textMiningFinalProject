from transformers import pipeline

class summary():
  def __init__(self, modelObj="facebook/bart-large-cnn"):
    self.modelObj = modelObj

  # summary a text at minimum 30 words, and max is 70 words
  def summary_result(self, text):
    summarizer = pipeline("summarization", model=self.modelObj)
    return summarizer(text[:1000], max_length=70, min_length=30, do_sample=False)[0]['summary_text']


# def main():
#   testText = """TAIPEI (Taiwan News) — Taiwan’s gross domestic product (GDP) per capita exceeded South Korea’s in 2022 
#             for the first time in a decade due to consistent higher average growth, the Ministry of Economic Affairs (MOEA) said Friday (April 28).
#             The average GDP per person in Taiwan reached US$32,811 (NT$1 million) in Taiwan, while South Korea recorded US$32,237 for last year, 
#             per CNA. The growth of the semiconductor industry and the return of Taiwanese investors from overseas helped Taiwan achieve an 
#             average yearly GDP growth of 3.2% over the past decade, while South Korea suffered under a declining currency to book only 2.6% growth per year.
#             The size of Taiwan’s manufacturing sector grew by a yearly average of 5.5% from 2013 to 2021, while in South Korea, manufacturing only 
#             expanded by an average of 2.8% per year during the same period. Exports also revealed a different pace of growth for the two countries, 
#             with an average annual growth rate of 4.6% for Taiwan and of 2.2% for South Korea, with the global average standing at 3%, according to MOEA data.
#             The gap in exports between the two has been narrowing over the past 10 years, as South Korea exported 1.8 times more than Taiwan in 2013, 
#             but 1.4 times more in 2021, the Economic Daily News reported."""

#   summarization = summary()
#   summary_result = summarization.summary_result(testText)
#   print (summary_result)

# main()