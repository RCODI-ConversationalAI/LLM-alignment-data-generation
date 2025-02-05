from pathlib import Path
import xml.etree.ElementTree as ET
import csv
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

def parse_emotion_lexicon(hierarchy_xml: Path, synset_xml: Path, output_csv: Path):
    """Optimized parser for emotion lexicon"""
    lemmatizer = WordNetLemmatizer()
    main_emotions = {'positive-emotion', 'negative-emotion', 'neutral-emotion', 'ambiguous-emotion'}
    entries = []
    
    # 1. Build hierarchy map
    print("Building hierarchy map...")
    hierarchy_tree = ET.parse(hierarchy_xml)
    parent_map = {}
    
    for categ in hierarchy_tree.findall('categ'):
        word = categ.get('name')
        parent = categ.get('isa')
        if parent:
            parent_map[word] = parent
    
    # Function to find emotion category
    def get_emotion_category(word):
        if word not in parent_map:
            return None
        
        path = []
        current = word
        while current in parent_map:
            path.append(current)
            current = parent_map[current]
            if current in main_emotions:
                # Return main emotion and immediate parent
                return (current, path[0])
        return None
    
    # 2. Process synsets
    print("Processing synsets...")
    
    # Clean and parse synset XML
    with open(synset_xml, 'r', encoding='utf-8') as f:
        content = f.read()
        if '<syn-list>' in content:
            content = content[content.index('<syn-list>'):]
    
    synset_tree = ET.fromstring(content)
    
    # Process each synset type
    pos_map = {
        'noun-syn': 'n',
        'adj-syn': 'a',
        'verb-syn': 'v',
        'adv-syn': 'r'
    }
    
    # Process entries
    for syn_type, pos in pos_map.items():
        for syn in synset_tree.findall(f'.//{syn_type}'):
            word = syn.get('categ')
            synset_id = syn.get('id')
            caus_stat = syn.get('caus-stat')
            
            # Get emotion category
            emotion_info = get_emotion_category(word)
            if emotion_info:
                main_cat, subcategory = emotion_info
                
                # Get correct lemma based on POS
                if pos == 'n':
                    lemma = lemmatizer.lemmatize(word.lower(), 'n')
                elif pos == 'v':
                    lemma = lemmatizer.lemmatize(word.lower(), 'v')
                elif pos == 'a':
                    lemma = lemmatizer.lemmatize(word.lower(), 'a')
                else:
                    lemma = word.lower()
                
                entries.append({
                    'main_category': main_cat,
                    'subcategory': subcategory,
                    'word': word,
                    'lemma': lemma,
                    'pos': pos,
                    'synset_id': synset_id,
                    'caus_stat': caus_stat if caus_stat else 'NA'
                })
    
    # 3. Write results
    print(f"Writing {len(entries)} entries to CSV...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, 
            fieldnames=['main_category', 'subcategory', 'word', 'lemma', 'pos', 'synset_id', 'caus_stat'])
        writer.writeheader()
        writer.writerows(entries)
    
    # Calculate statistics
    stats = defaultdict(lambda: defaultdict(int))
    for entry in entries:
        stats[entry['main_category']][entry['pos']] += 1
    
    return stats

def main():
    HOME_DIR = Path("/Volumes/ssd/01-ckj-postdoc/LLM-alignment-data-generation")
    hierarchy_xml = HOME_DIR / 'lexicon' / 'wn-domains-3.2' / 'wn-affect-1.1' / 'a-hierarchy.xml'
    synset_xml = HOME_DIR / 'lexicon' / 'wn-domains-3.2' / 'wn-affect-1.1' / 'a-synsets-30.xml'
    output_csv = HOME_DIR / 'lexicon' / 'affective_words.csv'
    
    print("Processing emotion lexicon...")
    stats = parse_emotion_lexicon(hierarchy_xml, synset_xml, output_csv)
    
    # Print detailed statistics
    print("\nEmotion Word Statistics by POS:")
    total = 0
    for main_cat, pos_counts in sorted(stats.items()):
        cat_total = sum(pos_counts.values())
        total += cat_total
        print(f"\n{main_cat} ({cat_total} total):")
        for pos, count in sorted(pos_counts.items()):
            print(f"  {pos}: {count} words")
    print(f"\nTotal entries: {total}")

if __name__ == "__main__":
    main()