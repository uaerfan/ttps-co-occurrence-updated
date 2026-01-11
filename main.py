from ast import Lambda
from audioop import reverse
from cProfile import label
from ctypes import util
from email import header
from itertools import count
from platform import node
import this
import pandas as pd 
import domain, utils
import tabulate as tb
from typing import Counter, List
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import networkx as nx 
import domain, utils
import statistics, math
import igraph
from networkx.algorithms import bipartite as bp
from networkx.algorithms import community as nxcm
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
# from apyori import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules
from prefixspan import PrefixSpan
import functools
from scipy.stats import pointbiserialr


tactics : List['domain.Tactic'] = []
techniques : List['domain.Technique'] = []
procedures  : List['domain.Procedure'] = []
groups  : List['domain.Group'] = []
softwares  : List['domain.Software'] = []
cocTTPs : List[List['domain.Technique']] = []

utils.buildDataSchema(tactics, techniques, procedures, groups, softwares)
cocGraph : nx.Graph = utils.initializeCocGraph(groups, softwares, cocTTPs, techniques, tactics)
cocDiGraph : nx.DiGraph =  utils.generateDiGraph2(cocTTPs, techniques, tactics)

# *** find top occurring techniques - Updated Code (Table III)
def findTopTenTechniques(techniques, cocTTPs):
    techniquesSortedBySupport = []
    
    # calculate support counts
    for te in techniques:
        count = 0
        for coc in cocTTPs:
            if te in coc: 
                count += 1
        techniquesSortedBySupport.append((te, count))
    
    # sort by support (high to low)
    techniquesSortedBySupport.sort(key=lambda v: v[1], reverse=True)

    # print top 10 techniques
    print("=== Top 10 Techniques ===")
    idx = 0
    for item in techniquesSortedBySupport[0:10]:
        idx += 1
        print(f"[{idx}] {item[0].id}: {item[0].name} @ {item[0].tactics[0].id}: {item[0].tactics[0].name} ==> {item[1]/len(cocTTPs):.2f}")
    
    # extract support values
    supports = [count / len(cocTTPs) for _, count in techniquesSortedBySupport]

    print("\nDEBUG: Sample supports:", supports[:10])  # 👈 debug line

    # classify supports
    support_gt_01 = [s for s in supports if s > 0.1]
    support_between_005_01 = [s for s in supports if 0.05 < s < 0.1]
    support_between_001_005 = [s for s in supports if 0.01 < s < 0.05]
    support_lt_001 = [s for s in supports if s < 0.01]

    # print summary
    print("\n=== Summary Statistics ===")
    print(f"Total techniques on support > 0.1: {len(support_gt_01)}")
    print(f"Mean support of techniques on support > 0.1: {np.mean(support_gt_01):.2f}" if support_gt_01 else "Mean support of techniques on support > 0.1: N/A")
    print(f"Median support of techniques on support > 0.1: {np.median(support_gt_01):.2f}" if support_gt_01 else "Median support of techniques on support > 0.1: N/A")
    print(f"Total techniques on 0.05 < support < 0.1: {len(support_between_005_01)}")
    print(f"Total techniques on 0.01 < support < 0.05: {len(support_between_001_005)}")
    print(f"Total techniques on support < 0.01: {len(support_lt_001)}")
    print(f"Total techniques: {len(techniquesSortedBySupport)}")

    return



# *** find top occurring techniques (Table III)

"""
def findTopTenTechniques(techniques, cocTTPs):
    techniquesSortedBySupport = []
    for te in techniques:
        sum = 0
        for coc in cocTTPs:
            if te in coc: 
                sum += 1
        techniquesSortedBySupport.append((te, sum))

    techniquesSortedBySupport.sort(key = lambda v : v[1], reverse=True)

    idx = 0
    for item in techniquesSortedBySupport[0:15]:
        idx += 1
        print(f"[{idx}] {item[0].id}: {item[0].name} @ {item[0].tactics[0].id}: {item[0].tactics[0].name} ==> {item[1]/len(cocTTPs)}")
    return

"""  

# *** find top tactics (Table IV)
def findTopTactics(techniques, tactics, cocTTPs):
    techniquesSortedBySupport = []
    for te in techniques:
        sum = 0
        for coc in cocTTPs:
            if te in coc: 
                sum += 1
        techniquesSortedBySupport.append((te, sum))

    techniquesSortedBySupport.sort(key = lambda v : v[1], reverse=True)

    idx = 0
    for item in techniquesSortedBySupport[0:10]:
        idx += 1
        print(f"[{idx}] {item[0].id}: {item[0].name} @ {item[0].tactics[0].id}: {item[0].tactics[0].name} ==> {item[1]/len(cocTTPs)}")
    
    
    topFourteenTechniques = []
    topTechniquesWithMinSpprt = [x[0] for x in techniquesSortedBySupport if x[1] > 59.9]

    columns = ['support', 'count', 'min', 'avg', 'med', 'stdev', 'max', 'top']
    index = [ta.id + ': ' + ta.name for ta in tactics]
    df = pd.DataFrame(index=index, columns=columns)

    for ta in tactics:
        topTechInThisTa = None
        maxValue = -1
        values = []
        support = 0
        for coc in cocTTPs:
            if ta in [x.tactics[0] for x in coc]:
                support += 1
        for te in topTechniquesWithMinSpprt:
            if te.tactics[0] == ta:
                value = [x[1] for x in techniquesSortedBySupport if x[0] == te][0]
                values.append(value/len(cocTTPs))
                if value > maxValue:
                    topTechInThisTa = te
                    maxValue = value
        if len(values) > 0:
            if len(values) > 1:
                topFourteenTechniques.append(topTechInThisTa)
                df.loc[f"{ta.id}: {ta.name}"] = [(support/len(cocTTPs)), (len(values)), (min(values)), (statistics.mean(values)), (statistics.median(values)), (statistics.stdev(values)), (max(values)), (topTechInThisTa.id + ': ' + topTechInThisTa.name)]
            if len(values) == 1:
                topFourteenTechniques.append(topTechInThisTa)
                df.loc[f"{ta.id}: {ta.name}"] = [(support/len(cocTTPs)), (len(values)), (min(values)), (statistics.mean(values)), (statistics.median(values)), (0), (max(values)), (topTechInThisTa.id + ': ' + topTechInThisTa.name)]

    for cols in columns[:-1]:
        df[cols] = df[cols].astype(float)
    df = df.round(2)
    print(tb.tabulate(df.sort_values(by='support', ascending=False), headers='keys', tablefmt='psql'))
    return
"""
# top ten combinations (Table V)
def getTopTenCombinations(cocTTPs):
    rules = utils.generateRules(cocTTPs)

    rules['alen'] = rules['antecedents'].apply(lambda x : len(x)) 
    rules['clen'] = rules['consequents'].apply(lambda x : len(x)) 
    dfq = rules.query("alen + clen == 2")

    print(dfq[['antecedents', 'consequents', 'support', 'confidence']].sort_values(by='support', ascending=False).head(20))
    return
"""
# top ten combinations (Table V - Updated)
def getTopTenCombinations(cocTTPs):
    rules = utils.generateRules(cocTTPs)

    # calculate antecedent and consequent lengths
    rules['alen'] = rules['antecedents'].apply(lambda x: len(x)) 
    rules['clen'] = rules['consequents'].apply(lambda x: len(x)) 

    # only pairs (2 techniques total)
    dfq = rules.query("alen + clen == 2")

    # top 10 by support
    top10 = dfq[['antecedents', 'consequents', 'support', 'confidence']] \
                .sort_values(by='support', ascending=False) \
                .head(10)

    # ---- summary stats ----
    total_sets = len(dfq)

    # flatten all antecedents + consequents into one set of techniques
    unique_techniques = set()
    for ant, con in zip(dfq['antecedents'], dfq['consequents']):
        unique_techniques.update(list(ant))
        unique_techniques.update(list(con))
    unique_count = len(unique_techniques)

    mean_support = dfq['support'].mean()
    median_support = dfq['support'].median()

    # print results
    print("\nTop 10 Combinations (by support):")
    print(top10)

    print("\n--- Summary ---")
    print(f"Total sets: {total_sets}")
    print(f"Unique Techniques: {unique_count}")
    print(f"Mean: {mean_support:.2f}")
    print(f"Median: {median_support:.2f}")

    return top10, total_sets, unique_count, mean_support, median_support

"""
# get top ten simple rules (Table VI)
def getTopTenSimpleRules(cocTTPs):
    rules = utils.generateRules(cocTTPs)

    rules['alen'] = rules['antecedents'].apply(lambda x : len(x)) 
    rules['clen'] = rules['consequents'].apply(lambda x : len(x)) 
    dfq = rules.query("alen + clen == 2")

    print(dfq[['antecedents', 'consequents', 'support', 'confidence']].sort_values(by='confidence', ascending=False).head(20))
    return
"""

# get top ten simple rules (Table VI - Updated)
def getTopTenSimpleRules(cocTTPs):
    rules = utils.generateRules(cocTTPs)

    # antecedent/consequent lengths
    rules['alen'] = rules['antecedents'].apply(lambda x: len(x)) 
    rules['clen'] = rules['consequents'].apply(lambda x: len(x)) 
    
    # filter: only simple rules with 1 antecedent + 1 consequent
    dfq = rules.query("alen + clen == 2")

    # --- Stats you asked for ---
    total_rules = len(dfq)
    min_conf = dfq['confidence'].min()
    mean_conf = dfq['confidence'].mean()
    median_conf = dfq['confidence'].median()

    print("=== Simple Rules Statistics ===")
    print(f"Total: {total_rules}")
    print(f"Minimum confidence: {min_conf:.2f}")
    print(f"Mean confidence: {mean_conf:.2f}")
    print(f"Median confidence: {median_conf:.2f}\n")

    # --- Top rules ---
    top_rules = dfq[['antecedents', 'consequents', 'support', 'confidence']] \
                    .sort_values(by='confidence', ascending=False) \
                    .head(20)

    print("=== Top Simple Rules (by confidence) ===")
    print(top_rules)

    return dfq, top_rules, {
        "total": total_rules,
        "min_conf": min_conf,
        "mean_conf": mean_conf,
        "median_conf": median_conf
    }
    
"""   
# get top ten compound rules (Table VII)
def getTopTenCompoundRules(cocTTPs):
    rules = utils.generateRules(cocTTPs)

    rules['alen'] = rules['antecedents'].apply(lambda x : len(x)) 
    rules['clen'] = rules['consequents'].apply(lambda x : len(x)) 
    dfq = rules.query("alen + clen > 2")

    print(dfq[['antecedents', 'consequents', 'support', 'confidence']].sort_values(by='confidence', ascending=False).head(20))
    return

"""
# get top ten compound rules (Table VII - Updated)
def getTopTenCompoundRules(cocTTPs):
    rules = utils.generateRules(cocTTPs)

    # antecedent & consequent lengths
    rules['alen'] = rules['antecedents'].apply(lambda x: len(x))
    rules['clen'] = rules['consequents'].apply(lambda x: len(x))

    # compound rules filter
    dfq = rules.query("alen + clen > 2")

    # print top 20 compound rules sorted by confidence
    print("\nTop 20 Compound Rules:")
    print(dfq[['antecedents', 'consequents', 'support', 'confidence']]
          .sort_values(by='confidence', ascending=False).head(20))

    # calculate summary stats
    total = len(dfq)
    min_conf = dfq['confidence'].min()
    mean_conf = dfq['confidence'].mean()
    median_conf = dfq['confidence'].median()

    print(f"\nSummary for Compound Rules:")
    print(f"Total: {total}")
    print(f"Minimum confidence: {min_conf:.2f}")
    print(f"Mean confidence: {mean_conf:.2f}")
    print(f"Median confidence: {median_conf:.2f}")

    return dfq
    
    
# get the adjacency matrix of the co-occurrence network (table VIII)
def getAdjacencyMatrix(cocDiGraph, techniques):
    print('*** printing the adjacency matrix ***')
    # print(dod)
    nodelists = list(cocDiGraph.nodes())

    Index = nodelists
    Columns = nodelists

    IndexWithTTPsObject = []
    for idx in Index:
        te = next((x for x in techniques if x.id == idx), None)
        IndexWithTTPsObject.append(te)

    IndexWithTTPsObject.sort(key = lambda v : v.id)
    IndexWithTTPsObject.sort(key = lambda v : v.tactics[0].sequence)
    Index = [f'{x.id}' for x in IndexWithTTPsObject]
    Columns = [f'{x.id}' for x in IndexWithTTPsObject]

    # for item in IndexWithTTPsObject:
    #     print(f'{item.id}: {item.name} | {item.tactics[0].id}: {item.tactics[0].name}')

    df = pd.DataFrame(index=Index, columns=Columns)

    for ix in Index:
        for cl in Columns:
            if cocDiGraph.has_edge(ix, cl):
                df.at[ix, cl] = '*'
            else:
                df.at[ix, cl] = ' '

    print(tb.tabulate(df, headers='keys', showindex=True, tablefmt='psql'))



#inclusing closeness centrality (Table IX)
def getTechniqueCentrality(cocDiGraph : nx.DiGraph, techniques : List['domain.Technique'], tactics : List['domain.Tactic']):
    
    # 1. Calculate Centrality Measures
    dcofNodes = nx.degree_centrality(cocDiGraph)
    bcOfNodes = nx.betweenness_centrality(cocDiGraph, normalized=False)
    
    # NEW: Calculate Closeness Centrality
    # In-closeness (Standard) and Out-closeness (Calculated on Reversed Graph)
    ccInOfNodes = nx.closeness_centrality(cocDiGraph)
    ccOutOfNodes = nx.closeness_centrality(cocDiGraph.reverse())
    
    techniqueIds = []
    techniqueNames = []
    dcinvalues = []
    dcoutvalues = []
    ccinvalues = []  # Closeness In
    ccoutvalues = [] # Closeness Out
    bcvalues = []
    tacticsList = []
    
    techniuqesInGraph = [x for x in techniques if x.id in list(cocDiGraph.nodes())]
    
    for te in techniuqesInGraph:
        techniqueIds.append(te.id)
        techniqueNames.append(te.name)
        
        # Extract values from NetworkX dictionaries
        dcinvalues.append((cocDiGraph.in_degree[te.id]))
        dcoutvalues.append((cocDiGraph.out_degree[te.id]))
        ccinvalues.append(ccInOfNodes[te.id])
        ccoutvalues.append(ccOutOfNodes[te.id])
        bcvalues.append(bcOfNodes[te.id])
        
        ta = cocDiGraph.nodes[te.id]['tactic']
        tacticsList.append([x.name for x in tactics if x.id == ta][0])
        
    # 2. Construct DataFrame
    data = {
        'id' : techniqueIds, 
        'Technique': techniqueNames, 
        'Tactic': tacticsList, 
        'IDC': dcinvalues,
        'ODC': dcoutvalues,  
        'BC': bcvalues, 
        'ICC': ccinvalues, # Incoming Closeness
        'OCC': ccoutvalues  # Outgoing Closeness
    }
    
    df = pd.DataFrame(data)
    df = df.round(2)
    
    # 3. Print Summary Statistics
    print(f"\n--- Centrality Distributions ---")
    print(f"IDC (In-Degree): Mean={statistics.mean(dcinvalues):.2f}, Quantiles={statistics.quantiles(dcinvalues, n=4)}")
    print(f"ODC (Out-Degree): Mean={statistics.mean(dcoutvalues):.2f}, Quantiles={statistics.quantiles(dcoutvalues, n=4)}")
    print(f"BC (Betweenness): Mean={statistics.mean(bcvalues):.2f}, Quantiles={statistics.quantiles(bcvalues, n=4)}")
    print(f"ICC (In-Closeness): Mean={statistics.mean(ccinvalues):.2f}, Quantiles={statistics.quantiles(ccinvalues, n=4)}")
    print(f"OCC (Out-Closeness): Mean={statistics.mean(ccoutvalues):.2f}, Quantiles={statistics.quantiles(ccoutvalues, n=4)}")
    
    print(f'\n*** Centrality of Techniques ***')
    # You can change sorting to 'ICC' or 'OCC' to see the most globally reachable nodes
    print(tb.tabulate(df.sort_values(by=['IDC'], ascending=False), headers='keys', showindex=False, tablefmt='psql'))
    
    # Correlation Matrix (Numeric Only)
    print("\n*** Correlation Matrix ***")
    numeric_df = df.select_dtypes(include=[np.number])
    print(tb.tabulate(numeric_df.corr().round(2), headers='keys', showindex=True, tablefmt='psql'))

    return df

"""
# get technique centrality measures (table IX)
def getTechniqueCentrality(cocDiGraph : nx.DiGraph, techniques : List['domain.Technique'], tactics : List['domain.Tactic']):
    
    dcofNodes = nx.degree_centrality(cocDiGraph)
    bcOfNodes = nx.betweenness_centrality(cocDiGraph, normalized = False)
    
    # pcInOfNodes = nx.katz_centrality(cocDiGraph, normalized = True)
    # pcOutOfNodes = nx.katz_centrality(cocDiGraph.reverse(), normalized = True)
    
    # ccInOfNodes = nx.closeness_centrality(cocDiGraph, wf_improved=True)
    # ccOutOfNodes = nx.closeness_centrality(cocDiGraph.reverse(), wf_improved=True)
    
    techniqueIds = []
    techniqueNames = []
    dcinvalues = []
    dcoutvalues = []
    # ccinvalues = []
    # ccoutvalues = []
    bcvalues = []
    # pcinvalues = []
    # pcoutvalues = []
    tacticsList = []
    
    techniuqesInGraph = [x for x in techniques if x.id in list(cocDiGraph.nodes())]
    
    for te in techniuqesInGraph:
        techniqueIds.append(te.id)
        techniqueNames.append(te.name)
        
        dcinvalues.append((cocDiGraph.in_degree[te.id]))
        dcoutvalues.append((cocDiGraph.out_degree[te.id]))
        # ccinvalues.append(ccInOfNodes[te.id])
        # ccoutvalues.append(ccOutOfNodes[te.id])
        bcvalues.append(bcOfNodes[te.id])
        # pcinvalues.append(pcInOfNodes[te.id])
        # pcoutvalues.append(pcOutOfNodes[te.id])
        
        # ta = next((x for x in tactics if x in te.tactics), None)
        # tacticsList.append(ta.id)
        
        ta = cocDiGraph.nodes[te.id]['tactic']
        tacticsList.append([x.name for x in tactics if x.id == ta][0])
        
    x=0
    data = {
        'id' : techniqueIds, 
        'Technique': techniqueNames, 
        'Tactic': tacticsList, 
        'IDC': dcinvalues,
        'ODC': dcoutvalues,  
        # 'bc': utils.normalizeList(bcvalues), 
        # 'cci': utils.normalizeList(ccinvalues),
        # 'cco': utils.normalizeList(ccoutvalues),
        # 'pci': utils.normalizeList(pcinvalues),
        # 'pco': utils.normalizeList(pcoutvalues)
        'BC': bcvalues, 
        # 'ICC': ccinvalues,
        # 'OCC': ccoutvalues,
        # 'IKC': pcinvalues,
        # 'OKC': pcoutvalues
    }
    
    df = pd.DataFrame(data)
    df = df.round(2)
    
    tacticnames = df['Tactic'].tolist()
    print(f"dci: {min(dcinvalues)} {statistics.mean(dcinvalues)} {statistics.quantiles(dcinvalues, n=4)}")
    print(f"dco: {min(dcoutvalues)} {statistics.mean(dcoutvalues)} {statistics.quantiles(dcoutvalues, n=4)}")
    print(f"bc: {min(bcvalues)} {statistics.mean(bcvalues)} {statistics.quantiles(bcvalues, n=4)}")
    # print(f"cci: {min(ccinvalues)} {statistics.mean(ccinvalues)} {statistics.quantiles(ccinvalues, n=4)}")
    # print(f"cco: {min(ccoutvalues)} {statistics.mean(ccoutvalues)} {statistics.quantiles(ccoutvalues, n=4)}")
    # print(f"kci: {min(pcinvalues)} {statistics.mean(pcinvalues)} {statistics.quantiles(pcinvalues, n=4)}")
    # print(f"kco: {min(pcoutvalues)} {statistics.mean(pcoutvalues)} {statistics.quantiles(pcoutvalues, n=4)}")
    
    print(f'*** centrality of techniques ***')
    print(tb.tabulate(df.sort_values(by=['IDC'], ascending=False).head(60), headers='keys', showindex=False, tablefmt='psql'))
    print(f'*** centrality of techniques stat ***')
    # print(tb.tabulate(df.describe(), headers='keys', tablefmt='psql'))
    
    # cclist = df['pc'].tolist()
    # print(np.percentile(cclist, 75))
    
    # print(tb.tabulate(df.corr().round(2),headers='keys', showindex=True, tablefmt='latex'))
    
    # dfm = pd.melt(df, id_vars=['id'], value_vars=['dc', 'cc', 'bc', 'pc'])
    # sns.violinplot(data=dfm, x='variable', y='value', inner='quartile',)
    # plt.show()
    
    
    dfq = df.query('Tactic == "TA0011"')
    # print(df.describe())
    
    ranges = []
    nodeCounts = []
    hueList = []
    
    bcvalues = utils.normalizeList(bcvalues)
    # ccinvalues = utils.normalizeList(ccinvalues)
    # pcinvalues = utils.normalizeList(pcinvalues)
    # ccoutvalues = utils.normalizeList(ccoutvalues)
    # pcoutvalues = utils.normalizeList(pcoutvalues)
    
    for i in range(0, 100, 20):
        nodeCount = len([x for x in bcvalues if x >= i/100 and x < (i+20)/100])
        ranges.append(f'{i/100}-{(i+20)/100}')
        hueList.append('BC')
        nodeCounts.append(nodeCount)
    return
"""
"""
def getTechniqueCentrality(cocDiGraph : nx.DiGraph, techniques : List['domain.Technique'], tactics : List['domain.Tactic']):
    
    # 1. Calculate Centralities
    dcofNodes = nx.degree_centrality(cocDiGraph)
    bcOfNodes = nx.betweenness_centrality(cocDiGraph, normalized = False)
    pcInOfNodes = nx.katz_centrality(cocDiGraph, normalized = True)
    pcOutOfNodes = nx.katz_centrality(cocDiGraph.reverse(), normalized = True)
    ccInOfNodes = nx.closeness_centrality(cocDiGraph, wf_improved=True)
    ccOutOfNodes = nx.closeness_centrality(cocDiGraph.reverse(), wf_improved=True)
    
    # 2. Initialize Lists
    lists = {
        'ids': [], 'names': [], 'dcin': [], 'dcout': [],
        'ccin': [], 'ccout': [], 'bc': [], 'pcin': [], 'pcout': [], 'tactics': []
    }
    
    techniuqesInGraph = [x for x in techniques if x.id in list(cocDiGraph.nodes())]
    
    for te in techniuqesInGraph:
        lists['ids'].append(te.id)
        lists['names'].append(te.name)
        lists['dcin'].append(cocDiGraph.in_degree[te.id])
        lists['dcout'].append(cocDiGraph.out_degree[te.id])
        lists['ccin'].append(ccInOfNodes[te.id])
        lists['ccout'].append(ccOutOfNodes[te.id])
        lists['bc'].append(bcOfNodes[te.id])
        lists['pcin'].append(pcInOfNodes[te.id])
        lists['pcout'].append(pcOutOfNodes[te.id])
        
        # Get Tactic Name (Fixed double-append)
        ta_id = cocDiGraph.nodes[te.id]['tactic']
        tactic_name = next((x.name for x in tactics if x.id == ta_id), "Unknown")
        lists['tactics'].append(tactic_name)
        
    # 3. Create DataFrame
    data = {
        'id' : lists['ids'], 
        'Technique': lists['names'], 
        'Tactic': lists['tactics'], 
        'IDC': lists['dcin'],
        'ODC': lists['dcout'],  
        'BC': lists['bc'], 
        'ICC': lists['ccin'],
        'OCC': lists['ccout'],
        'IKC': lists['pcin'],
        'OKC': lists['pcout']
    }
    
    df = pd.DataFrame(data)
    df = df.round(2)
    
    # Summary Prints
    print(f"IDC: {min(lists['dcin'])} {statistics.mean(lists['dcin'])} {statistics.quantiles(lists['dcin'], n=4)}")
    print(f"BC:  {min(lists['bc'])} {statistics.mean(lists['bc'])} {statistics.quantiles(lists['bc'], n=4)}")
    
    print(f'*** Centrality of Techniques ***')
    print(tb.tabulate(df.sort_values(by=['IDC'], ascending=False).head(60), headers='keys', showindex=False, tablefmt='psql'))
    
    # Correlation Matrix (Numeric Only)
    print("\n*** Correlation Matrix ***")
    numeric_df = df.select_dtypes(include=[np.number])
    print(tb.tabulate(numeric_df.corr().round(2), headers='keys', showindex=True, tablefmt='psql'))
    
    return df
"""
""" 
#rayhan vi implementation with clossness - error in the code

# get technique centrality measures (table IX)
def getTechniqueCentrality(cocDiGraph : nx.DiGraph, techniques : List['domain.Technique'], tactics : List['domain.Tactic']):
    
    dcofNodes = nx.degree_centrality(cocDiGraph)
    bcOfNodes = nx.betweenness_centrality(cocDiGraph, normalized = False)
    
    pcInOfNodes = nx.katz_centrality(cocDiGraph, normalized = True)
    pcOutOfNodes = nx.katz_centrality(cocDiGraph.reverse(), normalized = True)
    
    ccInOfNodes = nx.closeness_centrality(cocDiGraph, wf_improved=True)
    ccOutOfNodes = nx.closeness_centrality(cocDiGraph.reverse(), wf_improved=True)
    
    techniqueIds = []
    techniqueNames = []
    dcinvalues = []
    dcoutvalues = []
    ccinvalues = []
    ccoutvalues = []
    bcvalues = []
    pcinvalues = []
    pcoutvalues = []
    tacticsList = []
    
    techniuqesInGraph = [x for x in techniques if x.id in list(cocDiGraph.nodes())]
    
    for te in techniuqesInGraph:
        techniqueIds.append(te.id)
        techniqueNames.append(te.name)
        
        dcinvalues.append((cocDiGraph.in_degree[te.id]))
        dcoutvalues.append((cocDiGraph.out_degree[te.id]))
        ccinvalues.append(ccInOfNodes[te.id])
        ccoutvalues.append(ccOutOfNodes[te.id])
        bcvalues.append(bcOfNodes[te.id])
        pcinvalues.append(pcInOfNodes[te.id])
        pcoutvalues.append(pcOutOfNodes[te.id])
        
        ta = next((x for x in tactics if x in te.tactics), None)
        tacticsList.append(ta.id)
        
        ta = cocDiGraph.nodes[te.id]['tactic']
        tacticsList.append([x.name for x in tactics if x.id == ta][0])
        
    x=0
    data = {
        'id' : techniqueIds, 
        'Technique': techniqueNames, 
        'Tactic': tacticsList, 
        'IDC': dcinvalues,
        'ODC': dcoutvalues,  
        'bc': utils.normalizeList(bcvalues), 
        'cci': utils.normalizeList(ccinvalues),
        'cco': utils.normalizeList(ccoutvalues),
        'pci': utils.normalizeList(pcinvalues),
        'pco': utils.normalizeList(pcoutvalues),
        'BC': bcvalues, 
        'ICC': ccinvalues,
        'OCC': ccoutvalues,
        'IKC': pcinvalues,
        'OKC': pcoutvalues
    }
    
    df = pd.DataFrame(data)
    df = df.round(2)
    
    tacticnames = df['Tactic'].tolist()
    print(f"dci: {min(dcinvalues)} {statistics.mean(dcinvalues)} {statistics.quantiles(dcinvalues, n=4)}")
    print(f"dco: {min(dcoutvalues)} {statistics.mean(dcoutvalues)} {statistics.quantiles(dcoutvalues, n=4)}")
    print(f"bc: {min(bcvalues)} {statistics.mean(bcvalues)} {statistics.quantiles(bcvalues, n=4)}")
    print(f"cci: {min(ccinvalues)} {statistics.mean(ccinvalues)} {statistics.quantiles(ccinvalues, n=4)}")
    print(f"cco: {min(ccoutvalues)} {statistics.mean(ccoutvalues)} {statistics.quantiles(ccoutvalues, n=4)}")
    print(f"kci: {min(pcinvalues)} {statistics.mean(pcinvalues)} {statistics.quantiles(pcinvalues, n=4)}")
    print(f"kco: {min(pcoutvalues)} {statistics.mean(pcoutvalues)} {statistics.quantiles(pcoutvalues, n=4)}")
    
    print(f'*** centrality of techniques ***')
    print(tb.tabulate(df.sort_values(by=['IDC'], ascending=False).head(60), headers='keys', showindex=False, tablefmt='psql'))
    print(f'*** centrality of techniques stat ***')
    print(tb.tabulate(df.describe(), headers='keys', tablefmt='psql'))
    
    cclist = df['pc'].tolist()
    print(np.percentile(cclist, 75))
    
    print(tb.tabulate(df.corr().round(2),headers='keys', showindex=True, tablefmt='latex'))
    
    dfm = pd.melt(df, id_vars=['id'], value_vars=['dc', 'cc', 'bc', 'pc'])
    sns.violinplot(data=dfm, x='variable', y='value', inner='quartile',)
    plt.show()
    
    
    dfq = df.query('Tactic == "TA0011"')
    print(df.describe())
    
    ranges = []
    nodeCounts = []
    hueList = []
    
    bcvalues = utils.normalizeList(bcvalues)
    ccinvalues = utils.normalizeList(ccinvalues)
    pcinvalues = utils.normalizeList(pcinvalues)
    ccoutvalues = utils.normalizeList(ccoutvalues)
    pcoutvalues = utils.normalizeList(pcoutvalues)
    
    for i in range(0, 100, 20):
        nodeCount = len([x for x in bcvalues if x >= i/100 and x < (i+20)/100])
        ranges.append(f'{i/100}-{(i+20)/100}')
        hueList.append('BC')
        nodeCounts.append(nodeCount)
    return
"""

# --- Sensitivity Analysis (New Function - Table - XIV) ---
def performSensitivityAnalysis(cocTTPs: List[List['domain.Technique']]):
    print("\n" + "="*40)
    print("RUNNING TTP SENSITIVITY ANALYSIS")
    print("="*40)
    
    # Define ranges to test model stability
    #sup_range = [0.05, 0.08, 0.10, 0.12, 0.15]
    #conf_range = [0.10, 0.15, 0.20, 0.25, 0.30]

    sup_range = [0.05, 0.10, 0.15]
    conf_range = [0.10, 0.15, 0.20]
    
    results = []

    for s in sup_range:
        for c in conf_range:
            # Call generateRules with the specific sensitivity parameters
            # Note: We use the method from the utils module
            rules = utils.generateSensitivityRules(cocTTPs, minSup=s, minConf=c)
            
            num_rules = len(rules)
            
            # Check for Hub Stability: Is T1059 (Command/Scripting Interpreter) 
            # consistently appearing in the top 5 most frequent relationships?
            if not rules.empty:
                top_5 = rules.sort_values(by='support', ascending=False).head(5)
                # We check both antecedents and consequents for the Hub ID
                top_rule_check = top_5.apply(lambda row: 'T1059' in str(row['antecedents']) or 
                                                        'T1059' in str(row['consequents']), axis=1).any()
            else:
                top_rule_check = False
            
            results.append({
                'minSup': s,
                'minConf': c,
                'RuleCount': num_rules,
                'HubStable (T1059)': top_rule_check
            })

    # Convert to DataFrame for visualization and reporting
    df_sens = pd.DataFrame(results)
    
    # Display the results in a clear table
    print(tb.tabulate(df_sens, headers='keys', tablefmt='psql', showindex=False))
    
    # Stability Analysis Summary
    stable_count = df_sens['HubStable (T1059)'].sum()
    total_tests = len(df_sens)
    print(f"\n[Stability Check]")
    print(f"The primary Hub (T1059) remained a top-5 rule in {stable_count}/{total_tests} tests.")
    
    if stable_count / total_tests > 0.8:
        print("RESULT: Model is highly STABLE.")
    else:
        print("RESULT: Model is SENSITIVE to threshold changes. Consider lower Support.")

    return df_sens

# get Summary of the Dataset (Table II)
def summarizeDataset(groups, softwares):
    def stats_for_entities(entities):
        # number of techniques and tactics per entity
        techniques_counts = [len(set(p.techniques)) for p in entities]
        tactics_counts = [len(set([t.tactics[0] for t in set(p.techniques)])) for p in entities]

        # technique stats
        tech_avg = np.mean(techniques_counts)
        tech_med = np.median(techniques_counts)
        tech_min = np.min(techniques_counts)
        tech_max = np.max(techniques_counts)

        # tactic stats
        tac_avg = np.mean(tactics_counts)
        tac_med = np.median(tactics_counts)
        tac_min = np.min(tactics_counts)
        tac_max = np.max(tactics_counts)

        return {
            "count": len(entities),
            "tech": (tech_avg, tech_med, tech_min, tech_max),
            "tac": (tac_avg, tac_med, tac_min, tac_max),
        }

    # groups, malware(softwares), total
    group_stats = stats_for_entities(groups)
    malware_stats = stats_for_entities(softwares)
    total_stats = stats_for_entities(groups + softwares)

    # print in table style
    print("\nTABLE II. SUMMARY OF THE DATASET")
    print("Adversary    Count   Technique (Avg/Med, Min, Max)   Tactic (Avg/Med, Min, Max)")
    print(f"Groups      {group_stats['count']}   "
          f"{group_stats['tech'][0]:.1f} ({int(group_stats['tech'][1])})  {group_stats['tech'][2]}  {group_stats['tech'][3]}    "
          f"{group_stats['tac'][0]:.1f} ({int(group_stats['tac'][1])})  {group_stats['tac'][2]}  {group_stats['tac'][3]}")
    print(f"Malware     {malware_stats['count']}   "
          f"{malware_stats['tech'][0]:.1f} ({int(malware_stats['tech'][1])})  {malware_stats['tech'][2]}  {malware_stats['tech'][3]}    "
          f"{malware_stats['tac'][0]:.1f} ({int(malware_stats['tac'][1])})  {malware_stats['tac'][2]}  {malware_stats['tac'][3]}")
    print(f"Total       {total_stats['count']}   "
          f"{total_stats['tech'][0]:.1f} ({int(total_stats['tech'][1])})  {total_stats['tech'][2]}  {total_stats['tech'][3]}    "
          f"{total_stats['tac'][0]:.1f} ({int(total_stats['tac'][1])})  {total_stats['tac'][2]}  {total_stats['tac'][3]}")

    return group_stats, malware_stats, total_stats

summarizeDataset(groups, softwares)
findTopTenTechniques(techniques, cocTTPs)
findTopTactics(techniques, tactics, cocTTPs)
getTopTenCombinations(cocTTPs)
getTopTenSimpleRules(cocTTPs)
getTopTenCompoundRules(cocTTPs)
getAdjacencyMatrix(cocDiGraph, techniques)
getTechniqueCentrality(cocDiGraph, techniques, tactics)
performSensitivityAnalysis(cocTTPs)



# --- STRATIFIED ANALYSIS: GROUPS VS. SOFTWARE ---
# --- Section 7.4 ---

# A. Prepare the datasets
# Extract techniques lists specifically for Groups and specifically for Softwares
cocTTPs_groups = [list(set(g.techniques)) for g in groups]
cocTTPs_software = [list(set(s.techniques)) for s in softwares]

# Generate specific graphs for each stratum using your existing utils
cocDiGraph_groups = utils.generateDiGraph2(cocTTPs_groups, techniques, tactics)
cocDiGraph_software = utils.generateDiGraph2(cocTTPs_software, techniques, tactics)

datasets = [
    ("APT GROUPS ONLY", cocTTPs_groups, cocDiGraph_groups),
    ("MALWARE/SOFTWARE ONLY", cocTTPs_software, cocDiGraph_software)
]

# B. Run the Full Analysis Suite for each dataset
for label, current_ttp_list, current_graph in datasets:
    print("\n" + "="*80)
    print(f" STARTING STRATIFIED ANALYSIS: {label} ")
    print("="*80)
    
    # 1. Frequency Analysis (Techniques & Tactics)
    print(f"\n[1] Frequency Analysis for {label}")
    findTopTenTechniques(techniques, current_ttp_list)
    findTopTactics(techniques, tactics, current_ttp_list)
    
    # 2. Association Rule Mining (FP-Growth)
    print(f"\n[2] Association Rule Mining for {label}")
    getTopTenCombinations(current_ttp_list)
    getTopTenSimpleRules(current_ttp_list)
    getTopTenCompoundRules(current_ttp_list)
    
    # 3. Network Metrics (Centrality Measures)
    print(f"\n[3] Centrality Metrics for {label}")
    # This will output IDC, ODC, BC, ICC, and OCC for this specific stratum
    getTechniqueCentrality(current_graph, techniques, tactics)
    
    # 4. Structural Representation
    print(f"\n[4] Adjacency Matrix for {label}")
    getAdjacencyMatrix(current_graph, techniques)

    print("\n" + "-"*80)
    print(f" COMPLETED ANALYSIS FOR: {label} ")
    print("-"*80)

print("\nAll stratified analyses are complete. You can now compare the results side-by-side.")
