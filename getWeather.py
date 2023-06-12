# import required modules
import requests, json
import urllib
from bs4 import BeautifulSoup
from mtranslate import translate

class weather():
    def translate_text(self, text, target_language='en'):
        translated_text = translate(text, target_language)
        return translated_text

    def weather_today(self):
        # you can change region with city to make detail in specific city
        city_name = json.loads(urllib.request.urlopen('https://ipinfo.io').read().decode())['region']

        # creating url and requests instance
        url = "https://www.google.co.id/search?q=weather "+city_name
        html = requests.get(url).content
        
        # getting raw data
        soup = BeautifulSoup(html, 'html.parser')
        # print (soup)
        temp = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text
        str = soup.find('div', attrs={'class': 'BNeawe tAd8D AP7Wnd'}).text
        
        # formatting data
        data = str.split('\n')
        time = data[0]
        sky = data[1]
        
        # getting all div tag
        listdiv = soup.findAll('div', attrs={'class': 'BNeawe s3v9rd AP7Wnd'})
        strd = listdiv[5].text
        
        # getting other required data
        pos = strd.find('Wind')
        other_data = strd[pos:]
        
        # # printing all data
        # print("Temperature is", temp)
        # print("Time: ", translate_text(time))
        # print("Sky Description: ", translate_text(sky))
        return "I feel the temperature in " + city_name + " is " + self.translate_text(temp) + " and the sky is " + self.translate_text(sky)


# def main():
#     weatherToday = weather()
#     respond = weatherToday.weather_today()
#     print (respond)

# main()