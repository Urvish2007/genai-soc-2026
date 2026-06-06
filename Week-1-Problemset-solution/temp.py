import requests

url = "https://raw.githubusercontent.com/spyguessgame-boop/own_dataset/refs/heads/main/data.txt"

response = requests.get(url)
response.raise_for_status()

text_data = response.text

text_data = text_data[:1000]
print(text_data)

#THE 'BODY OF THE NATION'


# BUT the basin of the Mississippi is the Body of The Nation. All the other parts are but members, important in themselves, yet more important in their relations to this. Exclusive of the Lake basin and of 300,000 square miles in Texas and New Mexico, which in many aspects form a part of it, this basin contains about 1,250,000 square miles. In extent it is the second great valley of the world, being exceeded only by that of the Amazon. The valley of the frozen Obi approaches it in extent; that of La Plata comes next in space, and probably in habitable capacity, having about eight-ninths of its area; then comes that of the Yenisei, with about seven-ninths; the Lena, Amoor, Hoang-ho, Yang-tse-kiang, and Nile, five-ninths; the Ganges, less than one-half; the Indus, less than one-third; the Euphrates, one-fifth; the Rhine, one-fifteenth. It exceeds in extent the whole of Europe, exclusive of Russia, Norway, and Sweden. It would contain austria four times, germany 