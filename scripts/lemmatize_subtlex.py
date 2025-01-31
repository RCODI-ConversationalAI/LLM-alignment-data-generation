from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict
import csv
from nltk.stem import WordNetLemmatizer

def parse_emotion_hierarchy(xml_file):
    """Parse emotion hierarchy from XML"""
    print(f"Reading XML file: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Build basic hierarchy
    hierarchy = defaultdict(list)
    parents = {}
    
    print("Building hierarchy...")
    for categ in root.findall('categ'):
        name = categ.get('name')
        isa = categ.get('isa')
        if isa:
            hierarchy[isa].append(name)
            parents[name] = isa
    
    main_categories = hierarchy['emotion']
    return hierarchy, parents, main_categories

def extract_paths(main_category, hierarchy):
    """Extract paths efficiently using iteration"""
    paths = []
    stack = [(main_category, [main_category])]
    
    while stack:
        current, path = stack.pop()
        # If it's a leaf node
        if current not in hierarchy or not hierarchy[current]:
            paths.append(path)
        else:
            # Add children to stack
            for child in hierarchy[current]:
                new_path = path + [child]
                stack.append((child, new_path))
    
    return paths

def create_emotion_paths_csv(xml_file, output_file):
    """Create CSV with emotion hierarchy paths"""
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Parse hierarchy
    hierarchy, parents, main_categories = parse_emotion_hierarchy(xml_file)
    
    # Collect all paths
    print("Extracting paths...")
    all_paths = []
    for category in main_categories:
        print(f"Processing {category}...")
        paths = extract_paths(category, hierarchy)
        for path in paths:
            if len(path) >= 2:  # Ensure we have at least main category and word
                row = {
                    'main_category': path[0],
                    'subcategory': path[-2],
                    'word': path[-1],
                    'lemma': lemmatizer.lemmatize(path[-1].lower())
                }
                all_paths.append(row)
    
    # Write to CSV
    print(f"Writing to CSV: {output_file}")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['main_category', 'subcategory', 'word', 'lemma']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_paths)
    
    return len(all_paths)

if __name__ == "__main__":
    # Path settings
    HOME_DIR = Path("/Volumes/ssd/01-ckj-postdoc/LLM-alignment-data-generation")
    xml_path = HOME_DIR / 'lexicon' / 'wn-domains-3.2' / 'wn-affect-1.1' / 'a-hierarchy.xml'
    output_csv = HOME_DIR / 'lexicon' / 'emotion_paths.csv'
    
    print(f"Processing hierarchy from {xml_path}")
    print(f"Output will be saved to {output_csv}")
    
    # Create CSV
    total_paths = create_emotion_paths_csv(xml_path, output_csv)
    print(f"\nProcessed {total_paths} paths successfully!")
    
    # Print sample of results
    print("\nSample from the CSV:")
    with open(output_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < 5:  # Print first 5 rows
                print(f"{row['main_category']} -> {row['subcategory']} -> {row['word']} (lemma: {row['lemma']})")
            else:
                break