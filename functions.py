import numpy as np

def entropy(dset, target_col):
    """
    Entropy calculation based on given dataset.
    will have to carefully check the dataset to send to this function
    """
    # print(dset)
    # print(dset[target_col])
    elements, counts = np.unique(dset[target_col], return_counts=True)
    # print(elements, counts)
    
    entropy = np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    # print(entropy)
    
    return entropy


def infoGain(dset, target_col, feature):        
    """
    Information gain function.
    uses the entropy function as well.
    feature holds the remaining features not used in the tree yet by that particular 
    branch. so we will find only those features info gain from the dset given 
    """

    s_entropy = entropy(dset, target_col)
    s_total = len(dset.index)
    # print(s_total)
    
    elements, counts = np.unique(dset[feature], return_counts=True)
    # print(elements, counts)
    
    info = s_entropy
    for i in range(0, len(elements)):
        i_entropy = entropy(dset[dset[feature] == elements[i]], target_col)
        info -= ((counts[i] / s_total) * i_entropy)

    # print("info:", info)
    
    return info


def split_info(dataset, feature):     
    """
    Split information calculation
    # http://www.inf.unibz.it/dis/teaching/DWDM/slides2011/lesson5-Classification-2.pdf
    # https://www.saedsayad.com/decision_tree_super.htm
    Problem is it would still make bad decision on choosing ID, Day type attributes
    # http://www.ke.tu-darmstadt.de/lehre/archiv/ws0809/mldm/dt.pdf
    """

    elements, counts = np.unique(dataset[feature], return_counts=True)

    split_info_total = 0
    for i in range(0, len(elements)):
        i_split = - (counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
        split_info_total += i_split
    # print("split info:", split_info_total)
    
    return split_info_total


def bestclassifier(ds, target_col, remaining_col):
    """
    Finds best classifier in Greedy style
    whichever column has higher info gain, will get the best classifier tag
    """
    
    inf = {}
    for i in remaining_col:
        # print(i)
        inf[i] = infoGain(ds, target_col, i) / split_info(ds, i)

    info_gain_tup = list(inf.items())
    info_gain_list = list(inf.values())
    # print(info_gain_tup)
    # print()
    info_gain_list.sort(reverse=True)
    # print(info_gain_list[0])

    for i in info_gain_tup:
        if i[1] == info_gain_list[0]:
            best_classifier = i[0]
    # print('*******finding best classifier***********')
    # print(ds)
    # print('\nbest_classifier:', best_classifier)
    # print()
    
    return best_classifier


def ID3(main_dataset, dataset, target_col, attributes):
    """
    ID3 (Examples, Target_Attribute, Attributes)
        Create a root node for the tree
        If all examples are positive, Return the single-node tree Root, with label = +.
        If all examples are negative, Return the single-node tree Root, with label = -.
        If number of predicting attributes is empty, then Return the single node tree Root,
        with label = most common value of the target attribute in the examples.
        Otherwise Begin
            A ← The Attribute that best classifies examples.
            Decision Tree attribute for Root = A.
            For each possible value, vi, of A,
                Add a new tree branch below Root, corresponding to the test A = vi.
                Let Examples(vi) be the subset of examples that have the value vi for A
                If Examples(vi) is empty(**I did not check this at first**)(**later understood this will cause trouble!...**)
                    Then below this new branch add a leaf node with label = most common target value in the examples
                Else below this new branch add the subtree ID3 (Examples(vi), Target_Attribute, Attributes – {A})
        End
        Return Root
    """

    # Create a root node for the tree
    tree = {}
    tre = {}

    # If all examples are positive, Return the single-node tree Root, with label = +.
    # If all examples are negative, Return the single-node tree Root, with label = -.
    elements = np.unique(dataset[target_col])
    # print(attributes)
    
    if len(elements) == 1:
        # print('ALL pos/neg')
        # print(dataset)
        a = dataset[target_col].max()
        # print('->', a)
        # print('-------')
        # tree.append(a)
        return a

    # If number of predicting attributes is empty, then Return the single node tree Root,
    # with label = most common value of the target attribute in the examples.
    elif len(attributes) == 0:
        # print("No attribute left")
        # print(dataset[target_col].max())
        a = dataset[target_col].max()
        # tree.append(a)
        # print('==>>', a)
        return a

    else:
        node = bestclassifier(dataset, target_col, attributes)
        # tree['root'] = node
        # print('AND')
        children = np.unique(main_dataset[node])
        # print(np.unique(dataset[node]))

        new_att = np.setdiff1d(attributes, node)

        for child in children:

            # print('$$$$$$$$$$child$$$$$$$$$$$')
            # print('$$$$$$$$$', node, ' == ', child, '$$$$$$$$$')
            # print(child)
            data = dataset[dataset[node] == child]
            # print(data.head())

            if len(data) == 0:
                b = np.unique(dataset[target_col])[np.argmax(np.unique(dataset[target_col], return_counts=True)[1])]
                # print('!!0->', b)
                return b
            else:
                tre[child] = ID3(main_dataset, data, target_col, new_att)
        # print(tre)
        tree[node] = tre
    
    return tree


def predict(query, tree, default=1):
    """
    This function will predict a query data from a given decision tree
    """

    for key in list(query.keys()):
        if key in list(tree.keys()):
            # https://docs.python.org/3/tutorial/errors.html
            try:
                result = tree[key][query[key]]
            except:
                return default

            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result

