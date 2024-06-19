import re, string

x = '"[\'Following the record-breaking launch of NBA 2K16, the NBA 2K franchise continues to stake its claim as the most authentic sports video game with NBA 2K17. As the franchise that “all sports video games should aspire to be” (GamesRadar), NBA 2K17 will take the game to new heights and continue to blur the lines between video game and reality.\']"'
other_characters = '“”'

print(re.sub('[%s]' % re.escape(string.punctuation + other_characters), '' , x))