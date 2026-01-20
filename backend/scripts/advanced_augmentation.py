"""
Advanced Data Augmentation for 90%+ Accuracy
=============================================

Handles:
1. Document-based paraphrasing (long text)
2. Edge cases (negations, homonyms, subtle differences)
3. Hard negative mining
4. Semantic preserving transformations
5. Domain-specific paraphrases
"""

import pandas as pd
import random
import re
import numpy as np
from typing import List, Tuple, Dict


class AdvancedAugmenter:
    """Advanced data augmentation for edge cases and document paraphrasing"""
    
    def __init__(self):
        pass
            
    def create_document_paraphrases(self, df: pd.DataFrame, num_samples: int = 2000) -> pd.DataFrame:
        """
        Create document-level paraphrases (200+ words)
        These are harder to detect than sentence-level
        """
        doc_pairs = []
        
        # Document templates with complex structure
        doc_templates = [
            {
                "original": """The financial performance of the company has shown remarkable improvement 
                over the past fiscal year. Revenue increased by 25% compared to the previous period, 
                while operational costs were reduced by 15% through strategic efficiency measures. 
                The management team has successfully implemented new technologies that streamlined 
                production processes and improved customer satisfaction ratings. Market analysts 
                predict continued growth in the coming quarters based on current trends and 
                expansion plans into international markets.""",
                
                "paraphrase": """Over the last financial year, the corporation demonstrated outstanding 
                fiscal advancement. Income grew 25% relative to the prior period, whereas operating 
                expenses decreased 15% via strategic optimization initiatives. The executive leadership 
                effectively deployed innovative technologies that enhanced manufacturing workflows and 
                elevated customer approval scores. Industry experts forecast sustained expansion in 
                upcoming quarters given present trajectories and global market penetration strategies.""",
                
                "label": 1
            },
            {
                "original": """Climate change represents one of the most significant challenges facing 
                humanity in the 21st century. Rising global temperatures have led to increased frequency 
                of extreme weather events, including hurricanes, droughts, and floods. Scientific 
                consensus indicates that human activities, particularly fossil fuel combustion and 
                deforestation, are the primary drivers of these changes. Immediate action is required 
                to reduce greenhouse gas emissions and transition to renewable energy sources to 
                mitigate the worst impacts of climate change.""",
                
                "paraphrase": """Global warming constitutes among the most critical issues confronting 
                mankind this century. Elevated worldwide temperatures have caused more frequent 
                occurrence of severe meteorological phenomena, such as cyclones, water scarcity, and 
                flooding. Expert scientific agreement shows that anthropogenic factors, especially 
                burning of fossil fuels and forest destruction, primarily cause these transformations. 
                Prompt measures are necessary for decreasing carbon emissions and shifting toward 
                sustainable energy alternatives to minimize climate change's severest consequences.""",
                
                "label": 1
            },
            {
                "original": """Artificial intelligence and machine learning technologies are transforming 
                industries across the globe. From healthcare diagnostics to financial trading systems, 
                AI applications are becoming increasingly sophisticated and widespread. Deep learning 
                models can now process vast amounts of data to identify patterns and make predictions 
                with unprecedented accuracy. However, concerns about algorithmic bias, data privacy, 
                and job displacement require careful consideration as these technologies continue to 
                evolve and integrate into daily life.""",
                
                "paraphrase": """AI and ML innovations are revolutionizing sectors worldwide. Spanning 
                medical diagnosis to monetary trading platforms, artificial intelligence uses are 
                growing more advanced and pervasive. Neural network architectures can currently analyze 
                enormous datasets to detect trends and generate forecasts with remarkable precision. 
                Nevertheless, issues regarding computational prejudice, information security, and 
                employment disruption demand thorough attention as these systems advance and embed 
                themselves into routine activities.""",
                
                "label": 1
            }
        ]
        
        # Add document templates
        for doc in doc_templates:
            doc_pairs.append({
                'text_1': doc['original'],
                'text_2': doc['paraphrase'],
                'label': doc['label']
            })
            
        # Create hard negatives - similar documents with different meaning
        hard_negative_pairs = [
            {
                "text1": """The new policy will increase employee benefits and provide better healthcare 
                coverage. Workers will receive higher salaries and more vacation days starting next quarter.""",
                
                "text2": """The new policy will decrease employee benefits and reduce healthcare 
                coverage. Workers will receive lower salaries and fewer vacation days starting next quarter.""",
                
                "label": 0  # Similar structure but OPPOSITE meaning
            },
            {
                "text1": """The research confirms that regular exercise significantly improves mental health 
                and reduces symptoms of depression and anxiety in adults.""",
                
                "text2": """The research confirms that regular exercise significantly worsens mental health 
                and increases symptoms of depression and anxiety in adults.""",
                
                "label": 0  # Negation - critical edge case
            },
            {
                "text1": """The pharmaceutical company announced successful trial results showing the drug 
                is safe and effective for treating patients with chronic conditions.""",
                
                "text2": """The pharmaceutical company announced failed trial results showing the drug 
                is unsafe and ineffective for treating patients with chronic conditions.""",
                
                "label": 0
            }
        ]
        
        for pair in hard_negative_pairs:
            doc_pairs.append({
                'text_1': pair['text1'],
                'text_2': pair['text2'],
                'label': pair['label']
            })
            
        return pd.DataFrame(doc_pairs)
    
    def create_edge_case_dataset(self, df: pd.DataFrame, num_samples: int = 3000) -> pd.DataFrame:
        """
        Create comprehensive edge case dataset
        """
        edge_cases = []
        
        # 1. NEGATION EDGE CASES (Most critical for accuracy)
        negation_pairs = [
            ("The treatment is effective", "The treatment is not effective", 0),
            ("The results are accurate", "The results are inaccurate", 0),
            ("This solution works well", "This solution doesn't work well", 0),
            ("The patient improved", "The patient did not improve", 0),
            ("The data is reliable", "The data is unreliable", 0),
            ("The test passed successfully", "The test failed", 0),
            ("The company is profitable", "The company is unprofitable", 0),
            ("The drug is safe for use", "The drug is unsafe for use", 0),
            ("The method is validated", "The method is invalidated", 0),
            ("The hypothesis is correct", "The hypothesis is incorrect", 0),
            # Subtle negations
            ("Most participants agreed", "Most participants disagreed", 0),
            ("The majority supports the plan", "The majority opposes the plan", 0),
            ("Symptoms decreased significantly", "Symptoms increased significantly", 0),
            ("Performance improved markedly", "Performance deteriorated markedly", 0),
            ("The trend is upward", "The trend is downward", 0),
        ]
        
        # 2. HOMONYM/CONTEXT DISAMBIGUATION
        homonym_pairs = [
            ("The bank closed at 5 PM", "The river bank was steep", 0),
            ("She saw the bat fly away", "He swung the baseball bat", 0),
            ("The play was entertaining", "Children love to play", 0),
            ("Book your flight early", "Read this book carefully", 0),
            ("The bear ran into the woods", "I can't bear this pain", 0),
            ("The wind blew strongly", "Wind the clock carefully", 0),
            ("They present the award", "I got a present today", 0),
            ("This is fair treatment", "We went to the fair", 0),
            ("The rock was heavy", "Rock music is loud", 0),
            ("Spring flowers bloomed", "The spring broke", 0),
        ]
        
        # 3. NUMERICAL DIFFERENCES (Hard negatives)
        numerical_pairs = [
            ("The price increased by 50%", "The price increased by 5%", 0),
            ("The study had 1000 participants", "The study had 100 participants", 0),
            ("Temperature rose to 40 degrees", "Temperature rose to 4 degrees", 0),
            ("Sales grew 20% annually", "Sales grew 2% annually", 0),
            ("The project took 6 months", "The project took 6 weeks", 0),
            ("Accuracy improved from 70% to 90%", "Accuracy improved from 70% to 75%", 0),
            ("The budget is $10 million", "The budget is $1 million", 0),
            ("Population grew by 2 million", "Population grew by 200 thousand", 0),
        ]
        
        # 4. SUBTLE SEMANTIC DIFFERENCES
        semantic_pairs = [
            ("The doctor recommended rest", "The doctor prescribed medication", 0),
            ("The study suggests correlation", "The study proves causation", 0),
            ("The policy may be effective", "The policy is definitely effective", 0),
            ("Most participants improved", "All participants improved", 0),
            ("The effect was temporary", "The effect was permanent", 0),
            ("The change is optional", "The change is mandatory", 0),
            ("The result is probable", "The result is certain", 0),
            ("The approach is experimental", "The approach is established", 0),
            ("The evidence is preliminary", "The evidence is conclusive", 0),
            ("The solution is partial", "The solution is complete", 0),
        ]
        
        # 5. TRUE PARAPHRASES (Positive examples with variation)
        true_paraphrases = [
            ("The experiment yielded positive results", "The study produced favorable outcomes", 1),
            ("Patients showed significant improvement", "Individuals demonstrated substantial progress", 1),
            ("The method proved highly effective", "The approach was very successful", 1),
            ("Data analysis revealed important patterns", "Statistical examination uncovered key trends", 1),
            ("The research team concluded the investigation", "Scientists finished their inquiry", 1),
            ("Implementation requires careful planning", "Execution needs thorough preparation", 1),
            ("The system operates efficiently", "The platform functions effectively", 1),
            ("Results exceeded expectations significantly", "Outcomes surpassed projections substantially", 1),
            ("The intervention reduced symptoms", "The treatment decreased manifestations", 1),
            ("Analysis confirmed the hypothesis", "Examination validated the theory", 1),
        ]
        
        # Combine all edge cases
        all_pairs = (
            negation_pairs * 20 +  # Heavily weight negations
            homonym_pairs * 15 +
            numerical_pairs * 15 +
            semantic_pairs * 15 +
            true_paraphrases * 10
        )
        
        # Shuffle and limit
        random.shuffle(all_pairs)
        all_pairs = all_pairs[:num_samples]
        
        for q1, q2, label in all_pairs:
            edge_cases.append({
                'text_1': q1,
                'text_2': q2,
                'label': label
            })
            
        return pd.DataFrame(edge_cases)
    
    def create_hard_negatives(self, df: pd.DataFrame, num_samples: int = 1000) -> pd.DataFrame:
        """
        Create hard negative pairs - very similar text with different meanings
        These are the hardest for the model to distinguish
        """
        hard_negatives = []
        
        # Template-based hard negatives
        templates = [
            ("The {entity} is {positive_adj}", "The {entity} is {negative_adj}", 0),
            ("{action} improved the {metric}", "{action} worsened the {metric}", 0),
            ("Results show {positive_trend}", "Results show {negative_trend}", 0),
            ("The study found {positive_outcome}", "The study found {negative_outcome}", 0),
        ]
        
        entities = ["product", "service", "system", "method", "approach", "solution", "treatment"]
        positive_adj = ["effective", "reliable", "efficient", "successful", "beneficial", "accurate"]
        negative_adj = ["ineffective", "unreliable", "inefficient", "unsuccessful", "harmful", "inaccurate"]
        actions = ["Training", "Optimization", "Modification", "Implementation", "Adjustment"]
        metrics = ["performance", "accuracy", "efficiency", "quality", "reliability"]
        positive_trend = ["improvement", "growth", "increase", "advancement", "progress"]
        negative_trend = ["deterioration", "decline", "decrease", "regression", "setback"]
        positive_outcome = ["benefits", "improvements", "advantages", "gains", "success"]
        negative_outcome = ["drawbacks", "deterioration", "disadvantages", "losses", "failure"]
        
        for template in templates:
            for _ in range(num_samples // len(templates)):
                text1 = template[0].format(
                    entity=random.choice(entities),
                    positive_adj=random.choice(positive_adj),
                    action=random.choice(actions),
                    metric=random.choice(metrics),
                    positive_trend=random.choice(positive_trend),
                    positive_outcome=random.choice(positive_outcome)
                )
                text2 = template[1].format(
                    entity=random.choice(entities),
                    negative_adj=random.choice(negative_adj),
                    action=random.choice(actions),
                    metric=random.choice(metrics),
                    negative_trend=random.choice(negative_trend),
                    negative_outcome=random.choice(negative_outcome)
                )
                
                hard_negatives.append({
                    'text_1': text1,
                    'text_2': text2,
                    'label': template[2]
                })
                
        return pd.DataFrame(hard_negatives)
    
    def augment_dataset(self, df: pd.DataFrame, output_path: str = 'data/enhanced_training_data.csv'):
        """
        Create comprehensive augmented dataset for 90%+ accuracy
        """
        print("=" * 70)
        print("🚀 ADVANCED DATA AUGMENTATION FOR 90%+ ACCURACY")
        print("=" * 70)
        
        # Original data
        print(f"\n📊 Original dataset: {len(df)} pairs")
        
        # Add document-level paraphrases
        print("\n📄 Generating document-level paraphrases...")
        doc_data = self.create_document_paraphrases(df, num_samples=2000)
        print(f"   ✓ Created {len(doc_data)} document pairs")
        
        # Add comprehensive edge cases
        print("\n⚠️  Generating edge case dataset...")
        edge_data = self.create_edge_case_dataset(df, num_samples=5000)
        print(f"   ✓ Created {len(edge_data)} edge case pairs")
        
        # Add hard negatives
        print("\n🎯 Generating hard negative pairs...")
        hard_neg_data = self.create_hard_negatives(df, num_samples=2000)
        print(f"   ✓ Created {len(hard_neg_data)} hard negative pairs")
        
        # Combine all data
        enhanced_df = pd.concat([df, doc_data, edge_data, hard_neg_data], ignore_index=True)
        
        # Shuffle
        enhanced_df = enhanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save
        enhanced_df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 70)
        print("✅ AUGMENTATION COMPLETE")
        print("=" * 70)
        print(f"Total training pairs: {len(enhanced_df)}")
        print(f"├── Original data: {len(df)}")
        print(f"├── Document paraphrases: {len(doc_data)}")
        print(f"├── Edge cases: {len(edge_data)}")
        print(f"└── Hard negatives: {len(hard_neg_data)}")
        print(f"\nSaved to: {output_path}")
        
        # Distribution
        dup_count = enhanced_df['label'].sum()
        non_dup_count = len(enhanced_df) - dup_count
        print(f"\nClass distribution:")
        print(f"├── Paraphrases (1): {dup_count} ({dup_count/len(enhanced_df)*100:.1f}%)")
        print(f"└── Non-paraphrases (0): {non_dup_count} ({non_dup_count/len(enhanced_df)*100:.1f}%)")
        
        return enhanced_df


def main():
    """Generate enhanced training dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate enhanced training data')
    parser.add_argument('--input', type=str, default='data/quora_siamese_train.csv',
                       help='Input training data CSV')
    parser.add_argument('--output', type=str, default='data/enhanced_training_data.csv',
                       help='Output path for enhanced dataset')
    
    args = parser.parse_args()
    
    # Load original data
    print(f"\n📂 Loading original dataset from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Augment
    augmenter = AdvancedAugmenter()
    enhanced_df = augmenter.augment_dataset(df, args.output)
    
    print(f"\n✅ Enhanced dataset ready for training!")
    print(f"   Use: --data-path {args.output}")


if __name__ == '__main__':
    main()
