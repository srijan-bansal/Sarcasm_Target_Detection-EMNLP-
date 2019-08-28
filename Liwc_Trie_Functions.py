from Liwc_Trie import Liwc_Trie_Node

def create_trie(liwc_dict):
	T = Liwc_Trie_Node()
	for category, words in liwc_dict.items():
		# print category, "starting"
		for word in words:
			insert_word(T, word, category)
			# print "\t", word.encode("utf-8"), "inserted"
		# print category, "done"
	return T

def insert_word(T, word, category):
	if word[len(word) - 1] != "*":
		word = word + "$"
	t = T
	i = 0
	while i < len(word):
		match = False
		for child in t.children:
			if child.character == word[i]:
				match = True
				t = child
				break
		if match == False:
			break
		else:
			i = i + 1
	while i < len(word):
		child = Liwc_Trie_Node()
		child.character = word[i]
		t.children.append(child)
		t = child
		i = i + 1
	t.categories.add(category)

def get_liwc_categories(T, word):
	t = T
	categories = set([])
	i = 0
	word = word + "$"
	while i < len(word):
		match = False
		for child in t.children:
			if child.character == "*":
				categories = categories.union(child.categories)
			elif child.character == word[i]:
				match = True
				t = child
				break
		if match == False:
			break
		else:
			i = i + 1
	if i == len(word):
		categories = categories.union(t.categories)
	return categories
