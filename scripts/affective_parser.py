# This script is to extract affective words from WordNet and their frequency information from Subtlex-US.
# Updated on 04/15/2025
# If you have any questions, email to Cheonkam Jeong (cheonkamjeong@gmail.com)

import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from collections import defaultdict
from pathlib import Path
import pandas as pd
import nltk
import warnings
warnings.filterwarnings('ignore')

class EmotionLexiconBuilder:
    def __init__(self, hierarchy_file, synset_file):
        self.hierarchy = self._load_hierarchy(hierarchy_file)
        self.synsets = self._load_synsets(synset_file)
        self.emotion_words = defaultdict(set)
        self.processed_synsets = set()

    def _load_hierarchy(self, file_path):
        """Load and parse the emotion hierarchy"""
        try:
            tree = ET.parse(file_path)
            hierarchy = {}
            
            for categ in tree.findall('categ'):
                name = categ.get('name')
                parent = categ.get('isa')
                hierarchy[name] = {'parent': parent, 'children': set()}
                
            for name, info in hierarchy.items():
                if info['parent'] and info['parent'] in hierarchy:
                    hierarchy[info['parent']]['children'].add(name)
                    
            return hierarchy
        except Exception as e:
            print(f"Error loading hierarchy: {e}")
            return {}

    def _load_synsets(self, file_path):
        """Load synset mappings from XML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if '<syn-list>' in content:
                    content = content[content.index('<syn-list>'):]
            
            root = ET.fromstring(content)
            synsets = {}
            
            for syn_type in ['noun-syn', 'adj-syn', 'verb-syn', 'adv-syn']:
                for syn in root.findall(f'.//{syn_type}'):
                    synsets[syn.get('id')] = syn.get('categ')
                    
            return synsets
        except Exception as e:
            print(f"Error loading synsets: {e}")
            return {}

    def get_zipf_category(self, zipf_value):
        """
        Categorize words based on Zipf values (Van Heuven et al., 2014)
        1-3: Low-frequency words (≤1 per million)
        4-7: High-frequency words (≥10 per million)
        """
        try:
            zipf = float(zipf_value)
            if zipf <= 3:
                return "Low-frequency"
            elif zipf <= 7:
                return "High-frequency"
            else:
                return "Unknown"
        except:
            return "Unknown"

    def get_wordnet_words(self, word):
        """Get related words from WordNet for all parts of speech"""
        words = set()
        try:
            all_synsets = (
                wn.synsets(word, pos=wn.NOUN) +
                wn.synsets(word, pos=wn.ADJ) +
                wn.synsets(word, pos=wn.VERB) +
                wn.synsets(word, pos=wn.ADV)
            )
            
            for synset in all_synsets:
                if synset.name() in self.processed_synsets:
                    continue
                self.processed_synsets.add(synset.name())
                
                for lemma in synset.lemmas():
                    words.add(lemma.name().lower())
                    
                    for derived in lemma.derivationally_related_forms():
                        words.add(derived.name().lower())

                if synset.pos() == 'n':
                    for hyponym in synset.hyponyms():
                        words.update(l.name().lower() for l in hyponym.lemmas())
                        for h2 in hyponym.hyponyms():
                            words.update(l.name().lower() for l in h2.lemmas())
                    
                elif synset.pos() == 'v':
                    for troponym in synset.hyponyms():
                        words.update(l.name().lower() for l in troponym.lemmas())
                    
                elif synset.pos() in ['a', 's']:
                    for similar in synset.similar_tos():
                        words.update(l.name().lower() for l in similar.lemmas())
                        
                for lemma in synset.lemmas():
                    for antonym in lemma.antonyms():
                        words.add(antonym.name().lower())
                
        except Exception as e:
            print(f"Error processing word '{word}': {e}")
        
        return words

    def clean_word(self, word):
        """Clean a word by removing digits and converting underscores to spaces"""
        if isinstance(word, str):
            if not word.isdigit() and not any(c.isdigit() for c in word):
                cleaned = word.replace('_', ' ').lower().strip()
                if len(cleaned) > 2:
                    return cleaned
        return None

    def build_emotion_lexicon(self):
        """Build complete emotion lexicon"""
        basic_emotions = {
            'positive-emotion': [
                'joy', 'love', 'enthusiasm', 'happiness', 'delight',
                'amusement', 'elation', 'exultation', 'cheerfulness',
                'contentment', 'satisfaction', 'pleasure'
            ],
            'negative-emotion': [
                'sadness', 'fear', 'anger', 'anxiety', 'grief',
                'depression', 'rage', 'terror', 'disgust', 'shame'
            ],
            'neutral-emotion': [
                'surprise', 'apathy', 'calmness', 'contemplation'
            ],
            'ambiguous-emotion': [
                'awe', 'confusion', 'wonder', 'amazement'
            ]
        }
        
        for category, seeds in basic_emotions.items():
            print(f"\nProcessing category: {category}")
            hierarchy_words = self.expand_emotion_category(category)
            self.emotion_words[category].update(hierarchy_words)
            
            for seed in seeds:
                seed_words = self.get_wordnet_words(seed)
                clean_words = {self.clean_word(w) for w in seed_words}
                self.emotion_words[category].update(w for w in clean_words if w)
        
        return self.emotion_words

    def expand_emotion_category(self, category):
        """Expand an emotion category using WordNet-Affect and WordNet"""
        words = set()
        processed_categories = set()
    
        def process_category(cat_name):
            """Recursively process a category and its children"""
            if cat_name in processed_categories:
                return
            processed_categories.add(cat_name)
            
            # Add the category name itself if it's a valid word
            if cat_name and not cat_name.endswith('-emotion'):
                words.add(cat_name)
                words.update(self.get_wordnet_words(cat_name))
            
            # Process synsets for this category
            cat_synsets = [sid for sid, cat in self.synsets.items() if cat == cat_name]
            for synset_id in cat_synsets:
                try:
                    if '#' in synset_id:
                        word = synset_id.split('#')[1].lower()
                        if word and not word.isdigit():
                            words.add(word)
                            words.update(self.get_wordnet_words(word))
                except Exception as e:
                    print(f"Error processing synset {synset_id}: {e}")

            if cat_name in self.hierarchy:
                for child in self.hierarchy[cat_name]['children']:
                    process_category(child)
        
        process_category(category)

        return {w for w in words if self.clean_word(w)}

    def add_frequency_info(self, subtlex_file):
        """Add frequency information from SUBTLEX-US Excel file"""
        try:
            print("\nLoading SUBTLEX frequency data...")
            subtlex_df = pd.read_excel(subtlex_file, engine='openpyxl')
            
            enriched_lexicon = []
            total_words = sum(len(words) for words in self.emotion_words.values())
            processed = 0
            matched = 0
            
            freq_stats = {
                "Low-frequency": defaultdict(int),
                "High-frequency": defaultdict(int)
            }
            
            for category, words in self.emotion_words.items():
                print(f"\nProcessing frequency for {category}...")
                for word in words:
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Progress: {processed}/{total_words} words ({(processed/total_words)*100:.1f}%)")
                    
                    freq_info = subtlex_df[subtlex_df['Word'].str.lower() == word.lower()]
                    if not freq_info.empty:
                        matched += 1
                        zipf_value = freq_info['Zipf-value'].iloc[0]
                        freq_category = self.get_zipf_category(zipf_value)
                        freq_stats[freq_category][category] += 1
                        
                        entry = {
                            'category': category,
                            'word': word,
                            'zipf': zipf_value,
                            'frequency_category': freq_category,
                            'frequency': freq_info['FREQcount'].iloc[0],
                            'dominant_pos': freq_info['Dom_PoS_SUBTLEX'].iloc[0],
                            'pos_percentage': freq_info['Percentage_dom_PoS'].iloc[0],
                            'all_pos': freq_info['All_PoS_SUBTLEX'].iloc[0],
                            'cd_count': freq_info['CDcount'].iloc[0],
                            'cd_percentage': freq_info['SUBTLCD'].iloc[0]
                        }
                        enriched_lexicon.append(entry)
            
            # Print frequency statistics
            print(f"\nMatched {matched} out of {total_words} words")
            print("\nFrequency distribution by category:")
            for freq_cat, category_counts in freq_stats.items():
                print(f"\n{freq_cat}:")
                for emotion_cat, count in category_counts.items():
                    total_cat = sum(len(words) for cat, words in self.emotion_words.items() 
                                  if cat == emotion_cat)
                    percentage = (count / total_cat * 100) if total_cat > 0 else 0
                    print(f"  {emotion_cat}: {count} words ({percentage:.1f}%)")
            
            if enriched_lexicon:
                df = pd.DataFrame(enriched_lexicon)
                return df.sort_values(['category', 'zipf'], ascending=[True, False])
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error processing frequency information: {e}")
            return pd.DataFrame()

    def save_enriched_lexicon(self, df, output_file):
        """Save the frequency-enriched lexicon to CSV"""
        try:
            if df.empty:
                print("No data to save")
                return
            
            df.to_csv(output_file, index=False)
            
            base_path = output_file.parent
            base_name = output_file.stem
            
            for freq_cat in ['Low-frequency', 'High-frequency']:
                cat_df = df[df['frequency_category'] == freq_cat]
                if not cat_df.empty:
                    cat_file = base_path / f"{base_name}_{freq_cat.lower()}.csv"
                    cat_df.to_csv(cat_file, index=False)
            
            print(f"\nSaved lexicon to {output_file}")
            print(f"Total words with frequency information: {len(df)}")
            
            print("\nDistribution by category and frequency:")
            for category in df['category'].unique():
                cat_df = df[df['category'] == category]
                print(f"\n{category}:")
                freq_dist = cat_df['frequency_category'].value_counts()
                for freq_cat, count in freq_dist.items():
                    percentage = (count / len(cat_df)) * 100
                    print(f"  {freq_cat}: {count} words ({percentage:.1f}%)")

                high_freq = cat_df[cat_df['frequency_category'] == 'High-frequency'].nlargest(5, 'zipf')
                if not high_freq.empty:
                    print("\n  Top 5 high-frequency words:")
                    for _, row in high_freq.iterrows():
                        print(f"    {row['word']}: Zipf={row['zipf']:.2f}, "
                              f"POS={row['dominant_pos']}")
                        
        except Exception as e:
            print(f"Error saving enriched lexicon: {e}")

def main():
    try:
        try:
            import openpyxl
        except ImportError:
            print("Missing openpyxl package. Please install it using:")
            print("pip install openpyxl")
            return

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading WordNet...")
            nltk.download('wordnet')

        HOME_DIR = Path("/Volumes/ssd/01-ckj-postdoc/LLM-alignment-data-generation")
        hierarchy_xml = HOME_DIR / 'lexicon' / 'wn-domains-3.2' / 'wn-affect-1.1' / 'a-hierarchy.xml'
        synset_xml = HOME_DIR / 'lexicon' / 'wn-domains-3.2' / 'wn-affect-1.1' / 'a-synsets-30.xml'
        subtlex_file = HOME_DIR / 'lexicon' / 'subtlex-us' / 'SUBTLEX-US frequency list with PoS and Zipf information.xlsx'
        output_file = HOME_DIR / 'lexicon' / 'emotion_lexicon_with_freq.csv'

        for file_path in [hierarchy_xml, synset_xml, subtlex_file]:
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return

        print("Building emotion lexicon...")
        builder = EmotionLexiconBuilder(hierarchy_xml, synset_xml)
        emotion_lexicon = builder.build_emotion_lexicon()

        print("\nInitial lexicon statistics:")
        for category, words in emotion_lexicon.items():
            print(f"\n{category}: {len(words)} words")
            print("Sample words:", sorted(list(words))[:10])

        if subtlex_file.exists():
            print("\nAdding frequency information...")
            enriched_df = builder.add_frequency_info(subtlex_file)
            if not enriched_df.empty:
                builder.save_enriched_lexicon(enriched_df, output_file)
        else:
            print(f"\nSUBTLEX-US file not found at {subtlex_file}")

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()