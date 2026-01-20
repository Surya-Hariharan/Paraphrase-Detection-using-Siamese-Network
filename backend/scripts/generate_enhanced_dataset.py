"""
Enhanced Training Dataset Generator
====================================

Creates augmented training data to address model weaknesses:
1. Negation cases
2. Long text handling
3. Semantic/idiomatic paraphrases
4. Homonym disambiguation
5. Hard negative mining
"""

import pandas as pd
import random
from typing import List, Tuple
import re


class EnhancedDatasetGenerator:
    """Generate training data to fix model weaknesses"""
    
    def __init__(self):
        self.negation_prefixes = ['not', 'no', 'never', 'neither', 'none', 'nobody', 'nothing']
        self.negation_words = ['dis', 'un', 'im', 'in', 'non', 'anti']
        
    def create_negation_pairs(self, num_samples: int = 500) -> List[Tuple[str, str, int]]:
        """
        Generate negation training pairs.
        Label 0 = opposite meaning (should NOT be paraphrases)
        """
        base_sentences = [
            "The product is good quality",
            "I agree with your proposal",
            "The test was easy to complete",
            "This approach is possible",
            "The patient is stable",
            "The service is satisfactory",
            "He is honest in his dealings",
            "The results are accurate",
            "The process is efficient",
            "The treatment is effective",
            "The data is reliable",
            "The method is valid",
            "The outcome is positive",
            "The response is appropriate",
            "The solution is practical",
            "The design is functional",
            "The system is secure",
            "The performance is optimal",
            "The decision is correct",
            "The evidence is sufficient"
        ]
        
        pairs = []
        for sentence in base_sentences:
            # Add "not" negation
            negated = re.sub(r'\b(is|was|are|were)\b', r'\1 not', sentence)
            pairs.append((sentence, negated, 0))
            
            # Add prefix negation (un-, dis-, in-)
            if 'possible' in sentence:
                pairs.append((sentence, sentence.replace('possible', 'impossible'), 0))
            if 'honest' in sentence:
                pairs.append((sentence, sentence.replace('honest', 'dishonest'), 0))
            if 'accurate' in sentence:
                pairs.append((sentence, sentence.replace('accurate', 'inaccurate'), 0))
            if 'efficient' in sentence:
                pairs.append((sentence, sentence.replace('efficient', 'inefficient'), 0))
            if 'effective' in sentence:
                pairs.append((sentence, sentence.replace('effective', 'ineffective'), 0))
            if 'reliable' in sentence:
                pairs.append((sentence, sentence.replace('reliable', 'unreliable'), 0))
            if 'valid' in sentence:
                pairs.append((sentence, sentence.replace('valid', 'invalid'), 0))
            if 'appropriate' in sentence:
                pairs.append((sentence, sentence.replace('appropriate', 'inappropriate'), 0))
            if 'practical' in sentence:
                pairs.append((sentence, sentence.replace('practical', 'impractical'), 0))
            if 'functional' in sentence:
                pairs.append((sentence, sentence.replace('functional', 'dysfunctional'), 0))
            if 'secure' in sentence:
                pairs.append((sentence, sentence.replace('secure', 'insecure'), 0))
            if 'sufficient' in sentence:
                pairs.append((sentence, sentence.replace('sufficient', 'insufficient'), 0))
            if 'correct' in sentence:
                pairs.append((sentence, sentence.replace('correct', 'incorrect'), 0))
                
        return pairs[:num_samples]
    
    def create_homonym_pairs(self, num_samples: int = 300) -> List[Tuple[str, str, int]]:
        """
        Generate homonym disambiguation pairs.
        Label 0 = different meanings despite same word
        """
        homonym_contexts = [
            # Bank
            ("The bank was closed for the holiday", "He sat by the river bank fishing", 0),
            ("I need to deposit money at the bank", "The snow piled up on the bank", 0),
            
            # Bat
            ("The bat flew out of the cave", "He swung the baseball bat hard", 0),
            ("A vampire bat drinks blood", "The cricket bat needs repair", 0),
            
            # Patient
            ("The patient was admitted to the hospital", "The teacher was very patient with students", 0),
            ("The patient is recovering well", "You need to be patient while waiting", 0),
            
            # Match
            ("The tennis match was exciting", "He struck a match to light the candle", 0),
            ("The boxing match lasted 12 rounds", "The colors match perfectly", 0),
            
            # Can
            ("She can play the piano beautifully", "The soda can was recycled", 0),
            ("I can help you with that", "Open the can of beans", 0),
            
            # Bear
            ("The grizzly bear is dangerous", "I can't bear this pain anymore", 0),
            ("A polar bear lives in the Arctic", "Bear with me for a moment", 0),
            
            # Book
            ("I read an interesting book yesterday", "Please book the flight for tomorrow", 0),
            ("The book has 300 pages", "We need to book a hotel room", 0),
            
            # Light
            ("The room needs more light", "This box is very light in weight", 0),
            ("Turn on the light please", "He has a light workload today", 0),
            
            # Right
            ("Turn right at the intersection", "You have the right to remain silent", 0),
            ("The answer is right", "Raise your right hand", 0),
            
            # Wave
            ("She gave a friendly wave", "The ocean wave crashed on the shore", 0),
            ("Wave goodbye to your friends", "A heat wave is coming", 0),
            
            # Park
            ("Let's go to the park", "Park the car in the garage", 0),
            ("The national park is beautiful", "I can't park here legally", 0),
            
            # Watch
            ("I bought a new watch", "Let's watch a movie tonight", 0),
            ("My watch stopped working", "Watch out for that car", 0),
            
            # Spring
            ("Flowers bloom in spring", "The mattress has a broken spring", 0),
            ("Spring is my favorite season", "The cat will spring at the mouse", 0),
            
            # Fine
            ("The weather is fine today", "He paid a parking fine", 0),
            ("I'm feeling fine, thanks", "The fabric is very fine", 0),
            
            # Bark
            ("The dog's bark is loud", "Tree bark protects the trunk", 0),
            ("Don't bark at me like that", "The bark feels rough", 0),
        ]
        
        return homonym_contexts[:num_samples]
    
    def create_semantic_paraphrase_pairs(self, num_samples: int = 400) -> List[Tuple[str, str, int]]:
        """
        Generate semantic/idiomatic paraphrase pairs.
        Label 1 = same meaning despite different wording
        """
        semantic_pairs = [
            # Idioms
            ("Break a leg on your performance", "Good luck with your show", 1),
            ("It's raining cats and dogs", "It's raining very heavily", 1),
            ("Piece of cake for me", "This is very easy", 1),
            ("Hit the nail on the head", "You're exactly right", 1),
            ("Once in a blue moon", "Very rarely happens", 1),
            ("Costs an arm and a leg", "It's extremely expensive", 1),
            ("Spill the beans about it", "Reveal the secret", 1),
            ("Under the weather today", "Feeling sick", 1),
            ("The ball is in your court", "It's your decision now", 1),
            ("Bite the bullet", "Face the difficult situation", 1),
            
            # Euphemisms
            ("He passed away last night", "He died yesterday", 1),
            ("She's between jobs currently", "She is unemployed", 1),
            ("He's economically disadvantaged", "He is poor", 1),
            ("Let go from the company", "Fired from work", 1),
            ("Pre-owned vehicle for sale", "Used car available", 1),
            
            # Perspective changes
            ("The glass is half full", "The glass is half empty", 1),
            ("He's frugal with money", "He's careful with spending", 1),
            ("She's cautious and careful", "She's risk-averse", 1),
            
            # Different structures, same meaning
            ("The committee approved the proposal", "Approval was given to the proposal by the committee", 1),
            ("She opened the door", "The door was opened by her", 1),
            ("John gave Mary the book", "Mary received the book from John", 1),
            ("The chef prepared the meal", "The meal was prepared by the chef", 1),
            
            # Cause and effect
            ("Because of rain, the game was cancelled", "The game was cancelled due to rain", 1),
            ("The lack of rain caused drought", "Drought occurred because of insufficient rain", 1),
            
            # Synonymous expressions
            ("At the end of the day", "Ultimately speaking", 1),
            ("In my opinion", "From my perspective", 1),
            ("As a matter of fact", "Actually", 1),
            ("To make a long story short", "In brief", 1),
        ]
        
        return semantic_pairs[:num_samples]
    
    def create_long_text_pairs(self, num_samples: int = 200) -> List[Tuple[str, str, int]]:
        """
        Generate long text paraphrase pairs.
        """
        long_pairs = [
            # Technical paraphrases
            (
                "Machine learning is a subset of artificial intelligence that enables computer systems to "
                "learn from data and improve their performance over time without being explicitly programmed. "
                "It uses statistical techniques to give computers the ability to learn patterns from data, "
                "make predictions, and adapt to new information autonomously.",
                
                "ML, which falls under the AI umbrella, allows computational systems to enhance their capabilities "
                "through data analysis rather than traditional programming. By employing statistical methods, "
                "these systems can identify patterns, generate forecasts, and self-adjust based on new inputs.",
                1
            ),
            (
                "Climate change refers to long-term shifts in global weather patterns and average temperatures. "
                "The primary cause is human activities, particularly the burning of fossil fuels which releases "
                "greenhouse gases into the atmosphere. These gases trap heat and cause global temperatures to rise, "
                "leading to melting ice caps, rising sea levels, and more extreme weather events.",
                
                "Global climate shift describes sustained alterations in worldwide meteorological conditions and "
                "mean temperature levels. Anthropogenic factors, especially fossil fuel combustion releasing "
                "atmospheric greenhouse emissions, serve as the main driver. Such emissions capture thermal energy, "
                "elevating planetary temperatures and resulting in ice sheet dissolution, ocean level increase, "
                "and intensified climatic extremes.",
                1
            ),
            # Contrasting long texts (label 0)
            (
                "The stock market experienced significant volatility this week with technology stocks leading "
                "the decline. Investors are concerned about rising interest rates and potential recession. "
                "Major indices fell sharply as trading volume surged to record levels.",
                
                "Advances in renewable energy technology are transforming the power generation sector. Solar "
                "and wind installations have increased dramatically over the past decade. Government incentives "
                "and falling costs are driving widespread adoption of clean energy sources.",
                0
            ),
        ]
        
        return long_pairs[:num_samples]
    
    def create_hard_negative_pairs(self, num_samples: int = 300) -> List[Tuple[str, str, int]]:
        """
        Generate hard negative mining pairs.
        Similar words/structure but different meaning.
        """
        hard_negatives = [
            # Similar structure, different meaning
            ("The cat chased the mouse", "The mouse chased the cat", 0),
            ("John loves Mary", "Mary loves John", 0),  # Different relationships
            ("She sold him the car", "He sold her the car", 0),
            ("The teacher taught the student", "The student taught the teacher", 0),
            
            # Similar words, opposite actions
            ("He bought a new car", "He sold his new car", 0),
            ("She opened the window", "She closed the window", 0),
            ("The temperature increased", "The temperature decreased", 0),
            ("Profits rose this quarter", "Profits fell this quarter", 0),
            
            # Temporal opposites
            ("He arrived before the meeting", "He arrived after the meeting", 0),
            ("She woke up early", "She woke up late", 0),
            ("The store opens at 9 AM", "The store closes at 9 PM", 0),
            
            # Quantitative differences
            ("Most people agreed", "Few people agreed", 0),
            ("All students passed", "Some students passed", 0),
            ("He always arrives on time", "He never arrives on time", 0),
        ]
        
        return hard_negatives[:num_samples]
    
    def generate_enhanced_dataset(self, output_path: str = "data/enhanced_training_data.csv"):
        """
        Generate complete enhanced training dataset.
        """
        print("Generating Enhanced Training Dataset...")
        print("=" * 80)
        
        all_pairs = []
        
        # Generate each category
        print("1. Generating negation pairs...")
        negation_pairs = self.create_negation_pairs(500)
        all_pairs.extend(negation_pairs)
        print(f"   ✓ Added {len(negation_pairs)} negation pairs")
        
        print("2. Generating homonym pairs...")
        homonym_pairs = self.create_homonym_pairs(300)
        all_pairs.extend(homonym_pairs)
        print(f"   ✓ Added {len(homonym_pairs)} homonym pairs")
        
        print("3. Generating semantic/idiomatic pairs...")
        semantic_pairs = self.create_semantic_paraphrase_pairs(400)
        all_pairs.extend(semantic_pairs)
        print(f"   ✓ Added {len(semantic_pairs)} semantic pairs")
        
        print("4. Generating long text pairs...")
        long_pairs = self.create_long_text_pairs(200)
        all_pairs.extend(long_pairs)
        print(f"   ✓ Added {len(long_pairs)} long text pairs")
        
        print("5. Generating hard negative pairs...")
        hard_neg_pairs = self.create_hard_negative_pairs(300)
        all_pairs.extend(hard_neg_pairs)
        print(f"   ✓ Added {len(hard_neg_pairs)} hard negative pairs")
        
        # Create DataFrame
        df = pd.DataFrame(all_pairs, columns=['text_a', 'text_b', 'is_duplicate'])
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Enhanced dataset saved to: {output_path}")
        print(f"   Total pairs: {len(df)}")
        print(f"   Paraphrases: {df['is_duplicate'].sum()}")
        print(f"   Non-paraphrases: {len(df) - df['is_duplicate'].sum()}")
        print("=" * 80)
        
        return df


if __name__ == "__main__":
    generator = EnhancedDatasetGenerator()
    df = generator.generate_enhanced_dataset()
