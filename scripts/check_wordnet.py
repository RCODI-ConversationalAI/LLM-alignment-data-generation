from nltk.corpus import wordnet as wn
import nltk

def explore_emotion_word(word):
    synsets = wn.synsets(word, pos=wn.NOUN)
    
    print(f"\nExploring word: {word}")
    for synset in synsets:
        print("\nSynset:", synset.name())

        print("Definition:", synset.definition())

        if synset.examples():
            print("Examples:", synset.examples())
        
        hypernyms = synset.hypernyms()
        if hypernyms:
            print("Hypernyms:", [h.lemma_names() for h in hypernyms])
        
        hyponyms = synset.hyponyms()
        if hyponyms:
            print("Hyponyms:", [h.lemma_names() for h in hyponyms])
        
        if synset.pos() == 'a':
            similar = synset.similar_tos()
            if similar:
                print("Similar terms:", [s.lemma_names() for s in similar])

        print("Lemmas:", synset.lemma_names())

        first_lemma = synset.lemmas()[0]
        derived = first_lemma.derivationally_related_forms()
        if derived:
            print("Derivationally related forms:", 
                  [d.name() for d in derived])

print("=== Exploring 'joy' ===")
explore_emotion_word('joy')

print("\n=== Exploring 'happiness' ===")
explore_emotion_word('happiness')