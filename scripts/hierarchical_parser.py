from pathlib import Path
import xml.etree.ElementTree as ET
import csv
from nltk.stem import WordNetLemmatizer

def parse_emotion_lexicon(xml_path: Path, output_csv: Path):
    """
    Simple and fast parser for emotion lexicon with ~450 entries
    """
    # Initialize
    lemmatizer = WordNetLemmatizer()
    main_emotions = {'positive-emotion', 'negative-emotion', 'neutral-emotion', 'ambiguous-emotion'}
    
    # Parse XML in a single pass
    tree = ET.parse(xml_path)
    entries = []
    
    # Process each category
    for categ in tree.findall('categ'):
        word = categ.get('name')
        parent = categ.get('isa')
        
        # Skip root-level categories
        if parent == 'root' or parent is None:
            continue
            
        # If parent is a main emotion, add directly
        if parent in main_emotions:
            entries.append({
                'main_category': parent,
                'subcategory': parent,  # Use main category as subcategory for direct children
                'word': word,
                'lemma': lemmatizer.lemmatize(word.lower())
            })
        # If parent exists but isn't a main emotion, check its parent
        elif parent:
            grandparent = next((c.get('isa') for c in tree.findall('categ') 
                              if c.get('name') == parent), None)
            if grandparent in main_emotions:
                entries.append({
                    'main_category': grandparent,
                    'subcategory': parent,
                    'word': word,
                    'lemma': lemmatizer.lemmatize(word.lower())
                })
    
    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['main_category', 'subcategory', 'word', 'lemma'])
        writer.writeheader()
        writer.writerows(entries)
    
    # Calculate statistics
    stats = {}
    for entry in entries:
        main_cat = entry['main_category']
        stats[main_cat] = stats.get(main_cat, 0) + 1
    
    return stats

def main():
    # Paths
    HOME_DIR = Path("/Volumes/ssd/01-ckj-postdoc/LLM-alignment-data-generation")
    xml_path = HOME_DIR / 'lexicon' / 'wn-domains-3.2' / 'wn-affect-1.1' / 'a-hierarchy.xml'
    output_csv = HOME_DIR / 'lexicon' / 'emotion_paths.csv'
    
    # Process lexicon and get statistics
    print("Processing emotion lexicon...")
    stats = parse_emotion_lexicon(xml_path, output_csv)
    
    # Print results
    print("\nEmotion Word Statistics:")
    total = sum(stats.values())
    for category, count in sorted(stats.items()):
        print(f"{category}: {count} words ({count/total*100:.1f}%)")
    print(f"Total entries: {total}")

if __name__ == "__main__":
    main()