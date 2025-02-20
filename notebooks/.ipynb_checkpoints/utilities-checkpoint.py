from typing import List, Dict
import json

def load_jsonl(path:str)->List[Dict]:
    """loads a jsonlist file and returns a list of json objects"""
    output = []
    try:
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                output.append(data)
    except FileNotFoundError:
        print(f"Error: The file at {path} does not exist.")
        raise 
    return output


error_types = ['contradictory', 'relation', 'entity', 'invented', 'unverifiable', 'subjective']

############################ by passage ###########################


import re
from collections import Counter

def create_error_count_dict(text:str)-> Dict[str,int]:
    """
    Creates a dictionary of error type counts from annotations in the given text.

    Args:
        text (str): The text containing error annotations.

    Returns:
        dict: A dictionary with error types as keys and their counts as values.
    """
    # Define the error types you're interested in
    error_types = ['contradictory', 'relation', 'entity', 'invented', 'unverifiable', 'subjective']
    
    # Initialize a counter for each error type
    error_counts = Counter({error_type: 0 for error_type in error_types})

    # Iterate over each error type and count occurrences
    for error_type in error_types:
        # Regex pattern to find tags, case-insensitive
        pattern = fr'<{error_type}>.*?</{error_type}>'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        error_counts[error_type] += len(matches)
    
    return dict(error_counts)



def count_sentence_bound_errors(sentence:str, error_categories:List[str])->int:
    """returns the number of error annotations that start and end within the input sentence"""
    count = 0
    for category in error_categories:
        # Check for both opening and closing tags in the sentence
        if f"<{category}>" in sentence and f"</{category}>" in sentence:
            count += 1
    return count



def extract_gold_answer(text:str):
    """
    Remove and process annotations and return the unannotated gold answer
    Notes on annotations:
    - delete and mark tags are found within the error type tags
    """
    # pattern for error tag matches
    error_pattern = r'<(entity|contradictory|relation|invented|unverifiable|subjective|format|other|fictional)>(.*?)<\/\1>'  # Captures content inside these tags with group backreference

    def process_tags(match):
        """for each error pattern identified, process the delete and mark tags"""
        tag_name, text = match.groups()
        
        # process <delete> tags by removing
        text_without_deletes = re.sub(r'<delete>.*?<\/delete>', '', text)
        
        # process <mark> tags by returning tagged mark text
        mark_match = re.search(r'<mark>(.*?)<\/mark>', text_without_deletes)
        if mark_match:
            mark_text = mark_match.group(1)
            return mark_text
            
        return text_without_deletes

    # process text
    cleaned_text = re.sub(error_pattern, process_tags, text)
    
    return cleaned_text







def error_detected(agreement_gates:List[Dict])->bool:
    """identify from agreement gates if ANY error in text was detected"""
    # if no agreement gates then error was not detected
    if not agreement_gates:
        return False
        
    # look for open gate (meaning 'disagree' was detected)    
    error_detected = any(gate['is_open'] for gate in agreement_gates)
    if error_detected:
        return True
        
    return False

def count_errors(agreement_gates: List[Dict])-> int:
    """count the number of errors detected by agreement gates"""
    # if no agreement gates then error was not detected

    # if None or empty list
    if not agreement_gates:  
        return 0

    num_errors = sum(1 for gate in agreement_gates if gate['is_open'])
    return num_errors

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

def compare_stat_sig(dfA, dfB, label_col='has_error', pred_col='error_predicted', alpha=0.05):
    """
    Compare two models (represented by dfA and dfB) in terms of recall and precision
    using McNemar's test, and print the results.
    
    Parameters
    ----------
    dfA : pd.DataFrame
        DataFrame for Model A containing the ground truth and predictions.
    dfB : pd.DataFrame
        DataFrame for Model B containing the ground truth and predictions.
    label_col : str, optional
        Column name for ground truth labels (default 'has_error').
    pred_col : str, optional
        Column name for predicted labels (default 'error_predicted').
    alpha : float, optional
        Significance level for McNemar's test (default 0.05).

    Returns
    -------
    tuple
        A tuple containing the recall and precision contingency tables.
    """
    # ---------------------------
    # 1. Validate & combine data
    # ---------------------------
    if len(dfA) != len(dfB):
        raise ValueError("DataFrames dfA and dfB must have the same number of rows (same instances).")

    # Check that the ground truth is identical row-by-row
    if not dfA[label_col].equals(dfB[label_col]):
        raise ValueError("Ground-truth labels in dfA and dfB do not match row-by-row.")
    
    # Create a combined DataFrame with ground truth and both sets of predictions
    df = dfA.copy()
    df['predA'] = dfA[pred_col]
    df['predB'] = dfB[pred_col]
    
    # ---------------------------
    # 2. Compare RECALL (sensitivity)
    #    Only consider instances where ground truth is positive (== 1)
    # ---------------------------
    df_recall = df[df[label_col] == 1].copy()
    # Determine correctness for each model (True if prediction equals 1)
    df_recall['correctA'] = (df_recall['predA'] == 1)
    df_recall['correctB'] = (df_recall['predB'] == 1)
    
    # Build the 2x2 contingency table for recall
    cc = (df_recall['correctA'] & df_recall['correctB']).sum()  # both correct
    cw = (df_recall['correctA'] & (~df_recall['correctB'])).sum() # A correct, B wrong
    wc = ((~df_recall['correctA']) & df_recall['correctB']).sum() # A wrong,  B correct
    ww = ((~df_recall['correctA']) & (~df_recall['correctB'])).sum()# both wrong

    recall_table = [[cc, cw],
                    [wc, ww]]
    
    recall_result = mcnemar(recall_table, exact=False, correction=True)
    
    # ---------------------------
    # 3. Compare PRECISION
    #    Consider the union of predicted positives (A or B)
    # ---------------------------
    df_precision = df[(df['predA'] == 1) | (df['predB'] == 1)].copy()
    # For precision, correctness means: predicted positive and ground truth positive
    df_precision['correctA'] = (df_precision['predA'] == 1) & (df_precision[label_col] == 1)
    df_precision['correctB'] = (df_precision['predB'] == 1) & (df_precision[label_col] == 1)
    
    cc_p = (df_precision['correctA'] & df_precision['correctB']).sum()  # both correct
    cw_p = (df_precision['correctA'] & (~df_precision['correctB'])).sum() # A correct, B wrong
    wc_p = ((~df_precision['correctA']) & df_precision['correctB']).sum() # A wrong,  B correct
    ww_p = ((~df_precision['correctA']) & (~df_precision['correctB'])).sum()# both wrong
    
    precision_table = [[cc_p, cw_p],
                       [wc_p, ww_p]]
    
    precision_result = mcnemar(precision_table, exact=False, correction=True)
    
    # ---------------------------
    # 4. Print results
    # ---------------------------
    recall_stat = recall_result.statistic
    recall_pval = recall_result.pvalue
    recall_significant = (recall_pval < alpha)
    
    precision_stat = precision_result.statistic
    precision_pval = precision_result.pvalue
    precision_significant = (precision_pval < alpha)
    
    print("===== RECALL COMPARISON (McNemar's test) =====")
    print(f"Recall contingency table: {recall_table}")
    print(f"McNemar's statistic: {recall_stat:.4f}")
    print(f"p-value:             {recall_pval:.4g}")
    print(f"Significant at α={alpha}? {'YES' if recall_significant else 'NO'}")
    print()
    
    print("===== PRECISION COMPARISON (McNemar's test) =====")
    print(f"Precision contingency table: {precision_table}")
    print(f"McNemar's statistic: {precision_stat:.4f}")
    print(f"p-value:               {precision_pval:.4g}")
    print(f"Significant at α={alpha}? {'YES' if precision_significant else 'NO'}")
    
    return recall_table, precision_table



# def compare_stat_sig(dfA, dfB, label_col='has_error', pred_col='error_predicted', alpha=0.05):
#     """
#     Compare two models (represented by dfA and dfB) in terms of recall and precision
#     using McNemar's test, and print the results.

#     Parameters
#     ----------
#     dfA : pd.DataFrame
#         DataFrame for Model A containing the ground truth and predictions.
#     dfB : pd.DataFrame
#         DataFrame for Model B containing the ground truth and predictions.
#     label_col : str, optional
#         Column name for ground truth labels (default 'has_error').
#     pred_col : str, optional
#         Column name for predicted labels (default 'error_predicted').
#     alpha : float, optional
#         Significance level for McNemar's test (default 0.05).

#     Returns
#     -------
#     None
#         Prints the McNemar statistics and p-values for recall and precision
#         comparisons, along with significance flags.
#     """

#     # ---------------------------
#     # 1. Validate & extract data
#     # ---------------------------
#     if len(dfA) != len(dfB):
#         raise ValueError("DataFrames dfA and dfB must have the same number of rows (same instances).")

#     # Ground truth from each DataFrame
#     y_trueA = dfA[label_col].values
#     y_trueB = dfB[label_col].values
#     print(f"y true a")

#     # Ensure the ground truth is the same in both dataframes (if they must match exactly)
#     if not np.array_equal(y_trueA, y_trueB):
#         raise ValueError("Ground-truth labels in dfA and dfB do not match row-by-row.")
    
#     # We'll just use dfA's ground truth as the reference
#     y_true = y_trueA
    
#     # Predicted labels
#     y_predA_orig = dfA[pred_col].values
#     y_predB_orig = dfB[pred_col].values

    
#     # ---------------------------
#     # 2. Compare RECALL (sensitivity)
#     #    Only consider instances where y_true == 1
#     # ---------------------------
#     positives_mask = (y_true == 1)
#     print(f"positives mask: {len(positives_mask)}")
#     A_pred_recall = y_predA_orig[positives_mask]
#     B_pred_recall = y_predB_orig[positives_mask]
    
#     A_correct_recall = (A_pred_recall == 1).astype(int)
#     B_correct_recall = (B_pred_recall == 1).astype(int)

    
#     # Build the 2x2 contingency table for recall
#     cc = np.sum((A_correct_recall == 1) & (B_correct_recall == 1))  # both correct
#     cw = np.sum((A_correct_recall == 1) & (B_correct_recall == 0))  # A correct, B wrong
#     wc = np.sum((A_correct_recall == 0) & (B_correct_recall == 1))  # A wrong,  B correct
#     ww = np.sum((A_correct_recall == 0) & (B_correct_recall == 0))  # both wrong

#     recall_table = [[cc, cw],
#                     [wc, ww]]
    
#     recall_result = mcnemar(recall_table, exact=False, correction=True)
    
#     # ---------------------------
#     # 3. Compare PRECISION
#     #    Consider the union of predicted positives (A or B) 
#     #    to get a paired set for "predicted positive" comparison.
#     # ---------------------------
#     predicted_pos_union_mask = ((y_predA_orig == 1) | (y_predB_orig == 1))
#     A_pred_precision = y_predA_orig[predicted_pos_union_mask]
#     B_pred_precision = y_predB_orig[predicted_pos_union_mask]
#     y_true_precision = y_true[predicted_pos_union_mask]
    
#     A_correct_precision = ((A_pred_precision == 1) & (y_true_precision == 1)).astype(int)
#     B_correct_precision = ((B_pred_precision == 1) & (y_true_precision == 1)).astype(int)
    
#     # Build the 2x2 contingency table for precision
#     cc_p = np.sum((A_correct_precision == 1) & (B_correct_precision == 1))  # both correct
#     cw_p = np.sum((A_correct_precision == 1) & (B_correct_precision == 0))  # A correct, B wrong
#     wc_p = np.sum((A_correct_precision == 0) & (B_correct_precision == 1))  # A wrong,  B correct
#     ww_p = np.sum((A_correct_precision == 0) & (B_correct_precision == 0))  # both wrong
    
#     precision_table = [[cc_p, cw_p],
#                        [wc_p, ww_p]]
    
#     precision_result = mcnemar(precision_table, exact=False, correction=True)
    
#     # ---------------------------
#     # 4. Print results
#     # ---------------------------
#     recall_stat = recall_result.statistic
#     recall_pval = recall_result.pvalue
#     recall_significant = (recall_pval < alpha)
    
#     precision_stat = precision_result.statistic
#     precision_pval = precision_result.pvalue
#     precision_significant = (precision_pval < alpha)
    
#     print("===== RECALL COMPARISON (McNemar's test) =====")
#     print(f"Recall contingency table: {recall_table}")
#     print(f"McNemar's statistic: {recall_stat:.4f}")
#     print(f"p-value:             {recall_pval:.4g}")
#     print(f"Significant at α={alpha}? {'YES' if recall_significant else 'NO'}")
#     print()
    
#     print("===== PRECISION COMPARISON (McNemar's test) =====")
#     print(f"Precision contingency table: {precision_table}")
#     print(f"McNemar's statistic: {precision_stat:.4f}")
#     print(f"p-value:               {precision_pval:.4g}")
#     print(f"Significant at α={alpha}? {'YES' if precision_significant else 'NO'}")

#     return recall_table, precision_table
