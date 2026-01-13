"""
Comprehensive Evaluation Script for Siamese Network Architecture
=================================================================

Evaluates model performance on:
- Normal cases (standard paraphrases and non-paraphrases)
- Edge cases (empty text, special characters, very long/short text)
- False positive/negative cases
- Cross-domain generalization

Metrics Computed:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrix
- Error Analysis
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.model import TrainableSiameseModel


class SiameseEvaluator:
    """Comprehensive evaluator for Siamese Network"""
    
    def __init__(self, model_path: str, threshold: float = 0.8):
        """
        Initialize evaluator with trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            threshold: Classification threshold for paraphrase detection
        """
        self.model = TrainableSiameseModel(projection_dim=256, unfreeze_all=True)
        self.model.load_checkpoint(model_path)
        self.model.eval()
        self.threshold = threshold
        self.results = {}
        
    def create_test_cases(self) -> Dict[str, List[Tuple[str, str, int]]]:
        """
        Create comprehensive test cases covering various scenarios.
        
        Returns:
            Dictionary with test case categories
        """
        test_cases = {
            # Normal Paraphrases (True Positives Expected)
            'normal_paraphrases': [
                ("The cat is on the mat", "A cat sits on the mat", 1),
                ("I love programming in Python", "Python programming is my passion", 1),
                ("The movie was fantastic", "The film was amazing", 1),
                ("She bought a new car yesterday", "Yesterday, she purchased a new vehicle", 1),
                ("Climate change is a serious issue", "Global warming is a critical problem", 1),
                ("He runs every morning for exercise", "Every morning, he goes for a run to stay fit", 1),
                ("The restaurant serves delicious food", "This place has excellent cuisine", 1),
                ("Students need to study hard for exams", "Pupils must work diligently for their tests", 1),
                ("The company announced record profits", "The firm reported unprecedented earnings", 1),
                ("Technology is advancing rapidly", "Technological progress is accelerating quickly", 1),
            ],
            
            # Normal Non-Paraphrases (True Negatives Expected)
            'normal_non_paraphrases': [
                ("The cat is on the mat", "Dogs are loyal animals", 0),
                ("I love programming in Python", "Cooking pasta is an art", 0),
                ("The movie was fantastic", "Mathematics is challenging", 0),
                ("She bought a new car yesterday", "The sky is blue today", 0),
                ("Climate change is a serious issue", "I enjoy playing video games", 0),
                ("He runs every morning for exercise", "Pizza is my favorite food", 0),
                ("The restaurant serves delicious food", "Space exploration is expensive", 0),
                ("Students need to study hard for exams", "Flowers bloom in spring", 0),
                ("The company announced record profits", "Birds can fly south for winter", 0),
                ("Technology is advancing rapidly", "The ocean is very deep", 0),
            ],
            
            # Edge Cases - Similar Words, Different Meaning
            'false_positive_traps': [
                ("The bank was closed", "He sat on the river bank", 0),
                ("She can play the piano", "The can was empty", 0),
                ("The match started at noon", "He lit a match", 0),
                ("The bat flew away", "He swung the baseball bat", 0),
                ("The patient was treated", "The doctor was patient", 0),
                ("The bear was huge", "I can't bear this pain", 0),
                ("The book was interesting", "Book the flight now", 0),
                ("The wind was strong", "Wind the clock please", 0),
            ],
            
            # Edge Cases - Very Similar but Opposite Meaning
            'negation_cases': [
                ("The product is good", "The product is not good", 0),
                ("I agree with this", "I disagree with this", 0),
                ("He is honest", "He is dishonest", 0),
                ("This is possible", "This is impossible", 0),
                ("The test was easy", "The test was not easy", 0),
                ("She is happy", "She is unhappy", 0),
            ],
            
            # Edge Cases - Short Text
            'short_text_cases': [
                ("Yes", "Yeah", 1),
                ("No", "Nope", 1),
                ("Hello", "Hi", 1),
                ("Thanks", "Thank you", 1),
                ("OK", "Goodbye", 0),
                ("Yes", "No", 0),
            ],
            
            # Edge Cases - Very Long Text
            'long_text_cases': [
                (
                    "Artificial intelligence has revolutionized the way we interact with technology. "
                    "Machine learning algorithms can now process vast amounts of data and make predictions "
                    "with remarkable accuracy. Deep learning models, in particular, have shown exceptional "
                    "performance in tasks such as image recognition, natural language processing, and speech "
                    "recognition. The field continues to evolve rapidly with new architectures and techniques.",
                    
                    "The field of AI has transformed our relationship with digital systems. ML techniques "
                    "are capable of analyzing massive datasets and generating highly accurate forecasts. "
                    "Neural networks, especially deep learning systems, have demonstrated outstanding results "
                    "in areas like computer vision, language understanding, and voice recognition. This domain "
                    "is constantly advancing with innovative models and methodologies.",
                    1
                ),
                (
                    "Climate change is one of the most pressing challenges facing humanity. Rising global "
                    "temperatures, melting ice caps, and extreme weather events are becoming increasingly common.",
                    
                    "The stock market experienced significant volatility this week. Technology stocks led "
                    "the decline while energy sector showed resilience amid geopolitical tensions.",
                    0
                ),
            ],
            
            # Edge Cases - Special Characters and Numbers
            'special_cases': [
                ("The price is $100", "It costs 100 dollars", 1),
                ("Contact: john@email.com", "Email: john@email.com", 1),
                ("Call 555-1234", "Phone: 555-1234", 1),
                ("25% discount today!", "Today: 25% off", 1),
                ("ABC123", "XYZ789", 0),
                ("!!!URGENT!!!", "???HELP???", 0),
            ],
            
            # Semantic Paraphrases (Challenging Cases)
            'semantic_paraphrases': [
                ("The glass is half full", "The glass is half empty", 1),  # Same state, different perspective
                ("He kicked the bucket", "He passed away", 1),  # Idiom vs literal
                ("Break a leg!", "Good luck!", 1),  # Idiomatic expression
                ("It's raining cats and dogs", "It's raining heavily", 1),  # Idiom
                ("Piece of cake", "Very easy", 1),  # Idiom
            ],
            
            # Domain-Specific Cases
            'domain_specific': [
                # Medical
                ("The patient has hypertension", "The patient has high blood pressure", 1),
                ("MI occurred", "Myocardial infarction happened", 1),
                # Legal
                ("The defendant was acquitted", "The accused was found not guilty", 1),
                # Technical
                ("CPU utilization is high", "Processor usage is elevated", 1),
                ("RAM is insufficient", "Memory is inadequate", 1),
            ],
        }
        
        return test_cases
    
    def evaluate_test_set(self, test_cases: List[Tuple[str, str, int]]) -> Dict[str, Any]:
        """
        Evaluate model on a test set.
        
        Args:
            test_cases: List of (text_a, text_b, label) tuples
            
        Returns:
            Dictionary with predictions and ground truth
        """
        predictions = []
        probabilities = []
        ground_truth = []
        
        for text_a, text_b, label in test_cases:
            try:
                # Get similarity score
                result = self.model.calculate_similarity(text_a, text_b)
                cosine_sim = result['cosine_similarity']
                
                # Convert from [-1, 1] to [0, 1]
                similarity = (cosine_sim + 1) / 2
                
                # Binary prediction
                prediction = 1 if similarity >= self.threshold else 0
                
                predictions.append(prediction)
                probabilities.append(similarity)
                ground_truth.append(label)
                
            except Exception as e:
                print(f"Error processing pair: {str(e)}")
                # Default to negative case on error
                predictions.append(0)
                probabilities.append(0.0)
                ground_truth.append(label)
        
        return {
            'predictions': np.array(predictions),
            'probabilities': np.array(probabilities),
            'ground_truth': np.array(ground_truth),
            'test_cases': test_cases
        }
    
    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            results: Dictionary with predictions and ground truth
            
        Returns:
            Dictionary with metric scores
        """
        y_true = results['ground_truth']
        y_pred = results['predictions']
        y_prob = results['probabilities']
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # ROC-AUC and PR-AUC (only if we have both classes)
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            
            # Additional metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return metrics
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation on all test case categories.
        
        Returns:
            Comprehensive evaluation results
        """
        print("=" * 80)
        print("SIAMESE NETWORK COMPREHENSIVE EVALUATION")
        print("=" * 80)
        print(f"\nModel: Siamese SBERT + Projection Head")
        print(f"Threshold: {self.threshold}")
        print(f"Device: {self.model.device}")
        print(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        test_cases = self.create_test_cases()
        all_results = {}
        
        # Evaluate each category
        for category, cases in test_cases.items():
            print(f"\n{'=' * 80}")
            print(f"Category: {category.upper().replace('_', ' ')}")
            print(f"Test Cases: {len(cases)}")
            print("=" * 80)
            
            results = self.evaluate_test_set(cases)
            metrics = self.calculate_metrics(results)
            
            # Store results
            all_results[category] = {
                'results': results,
                'metrics': metrics,
                'num_cases': len(cases)
            }
            
            # Print metrics
            print(f"\n📊 METRICS:")
            print(f"  Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  Precision:  {metrics['precision']:.4f}")
            print(f"  Recall:     {metrics['recall']:.4f}")
            print(f"  F1-Score:   {metrics['f1_score']:.4f}")
            
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
                print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
            
            if 'true_positives' in metrics:
                print(f"\n📈 CONFUSION MATRIX:")
                print(f"  True Positives:  {metrics['true_positives']}")
                print(f"  True Negatives:  {metrics['true_negatives']}")
                print(f"  False Positives: {metrics['false_positives']}")
                print(f"  False Negatives: {metrics['false_negatives']}")
                print(f"  Sensitivity:     {metrics['sensitivity']:.4f}")
                print(f"  Specificity:     {metrics['specificity']:.4f}")
            
            # Error analysis
            errors = self._analyze_errors(results)
            if errors:
                print(f"\n❌ ERRORS ({len(errors)}):")
                for idx, error in enumerate(errors[:3], 1):  # Show first 3 errors
                    print(f"  {idx}. Predicted: {error['predicted']}, Actual: {error['actual']}")
                    print(f"     Text A: {error['text_a'][:50]}...")
                    print(f"     Text B: {error['text_b'][:50]}...")
                    print(f"     Similarity: {error['similarity']:.4f}")
        
        # Overall evaluation
        print(f"\n\n{'=' * 80}")
        print("OVERALL EVALUATION SUMMARY")
        print("=" * 80)
        
        overall_results = self._calculate_overall_metrics(all_results)
        
        print(f"\n📊 AGGREGATED METRICS:")
        print(f"  Overall Accuracy:  {overall_results['overall_accuracy']:.4f} ({overall_results['overall_accuracy']*100:.2f}%)")
        print(f"  Overall Precision: {overall_results['overall_precision']:.4f}")
        print(f"  Overall Recall:    {overall_results['overall_recall']:.4f}")
        print(f"  Overall F1-Score:  {overall_results['overall_f1']:.4f}")
        print(f"  Total Test Cases:  {overall_results['total_cases']}")
        print(f"  Correct Predictions: {overall_results['correct_predictions']}")
        print(f"  Incorrect Predictions: {overall_results['incorrect_predictions']}")
        
        print(f"\n📋 CATEGORY PERFORMANCE:")
        for category, data in all_results.items():
            acc = data['metrics']['accuracy']
            status = "✅" if acc >= 0.8 else "⚠️" if acc >= 0.6 else "❌"
            print(f"  {status} {category:30s} Accuracy: {acc:.4f} ({data['num_cases']} cases)")
        
        # Save results
        self._save_results(all_results, overall_results)
        
        return {
            'category_results': all_results,
            'overall_results': overall_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_errors(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze misclassified cases"""
        errors = []
        
        y_true = results['ground_truth']
        y_pred = results['predictions']
        y_prob = results['probabilities']
        test_cases = results['test_cases']
        
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                errors.append({
                    'text_a': test_cases[i][0],
                    'text_b': test_cases[i][1],
                    'actual': int(y_true[i]),
                    'predicted': int(y_pred[i]),
                    'similarity': float(y_prob[i])
                })
        
        return errors
    
    def _calculate_overall_metrics(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall metrics across all categories"""
        all_predictions = []
        all_ground_truth = []
        total_cases = 0
        
        for category, data in all_results.items():
            all_predictions.extend(data['results']['predictions'])
            all_ground_truth.extend(data['results']['ground_truth'])
            total_cases += data['num_cases']
        
        all_predictions = np.array(all_predictions)
        all_ground_truth = np.array(all_ground_truth)
        
        correct = np.sum(all_predictions == all_ground_truth)
        incorrect = len(all_predictions) - correct
        
        return {
            'overall_accuracy': accuracy_score(all_ground_truth, all_predictions),
            'overall_precision': precision_score(all_ground_truth, all_predictions, zero_division=0),
            'overall_recall': recall_score(all_ground_truth, all_predictions, zero_division=0),
            'overall_f1': f1_score(all_ground_truth, all_predictions, zero_division=0),
            'total_cases': total_cases,
            'correct_predictions': int(correct),
            'incorrect_predictions': int(incorrect)
        }
    
    def _save_results(self, all_results: Dict[str, Any], overall_results: Dict[str, Any]):
        """Save evaluation results to JSON file"""
        output_dir = Path(__file__).parent.parent / "evaluation_results"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"evaluation_{timestamp}.json"
        
        # Prepare serializable results
        serializable_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_path': str(self.model.SBERT_MODEL),
                'threshold': self.threshold,
                'device': str(self.model.device)
            },
            'overall_metrics': overall_results,
            'category_metrics': {}
        }
        
        for category, data in all_results.items():
            serializable_results['category_metrics'][category] = {
                'num_cases': data['num_cases'],
                'metrics': data['metrics']
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    """Main evaluation function"""
    # Path to trained model
    model_path = "../checkpoints/best_model.pt"
    
    # Initialize evaluator
    evaluator = SiameseEvaluator(model_path, threshold=0.8)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
