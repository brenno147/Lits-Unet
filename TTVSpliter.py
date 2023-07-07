import json
import random

pathDataset = "/home/brenno/convmixunet/LITS/Images/"
randomDataset = False

def defineTrainTestValidation(trainPercents,testPercents,validationPercents):

    def getListPatient(jsonFile):
        listPatient = []
        with open(jsonFile) as json_file:
            data = json.load(json_file)
            for p in data:
                listPatient.append(p['case_id'])
        return listPatient

    jsonFile = pathDataset +'json_file.json'


    listIDCases = getListPatient(jsonFile)
    sizeDataset = len(listIDCases)
    typeImages = ['Exam','Segmentation']

    ########################################### Train #################################################
    listCaseTrainExam = []
    listCaseTrainMasks = []

    sizeTrain = int((trainPercents/100) * sizeDataset)


    subListIDCasesTrain = []
    print("#####################################")
    print("Total List Case: ",len(listIDCases))
    print("#####################################\n")
    print("Size Train: ",sizeTrain)

    ### Random Dataset
    if randomDataset:
        for i in range(sizeTrain):
            case_id = random.choice(listIDCases)
            #print("Case Random",case)
            subListIDCasesTrain.append(case_id)
            listIDCases.remove(case_id)
    ### Without Random Pacients
    else:
        for i in range(sizeTrain):
            case_id = listIDCases[0]
            subListIDCasesTrain.append(case_id)
            #print(case)
            listIDCases.remove(case_id)

    #print(len(subListIDCasesTrain))
    for case_id in subListIDCasesTrain:

        #Exam Image
        pathImageExam   = pathDataset +  case_id + '/' + typeImages[0] + '/'
        #pathImageExam   = pathDataset +  case + '/'  + typeImages[0] + '/'
        listCaseTrainExam.append(pathImageExam)

        pathImageMasks   = pathDataset +  case_id + '/' +  typeImages[1] + '/'
        #pathImageMasks   = pathDataset +  case + '/'  + typeImages[1] + '/'
        listCaseTrainMasks.append(pathImageMasks)

    ########################################### Test #################################################
    listCaseTestExam = []
    listCaseTestMasks = []

    sizeTest = int((testPercents/100) * sizeDataset)

    subListIDCasesTest = []
    print("Current Size List Case: ", len(listIDCases))
    print("Size Test: ",sizeTest)

    ### Random Dataset
    if randomDataset:
        for i in range(sizeTest):
            case_id = random.choice(listIDCases)
            #print("Case Random",case)
            subListIDCasesTest.append(case_id)
            listIDCases.remove(case_id)
    ### Without Random Pacients
    else:
        for i in range(sizeTest):
            case_id = listIDCases[0]
            subListIDCasesTest.append(case_id)
            listIDCases.remove(case_id)

    #print(len(listIDCases))
    #print(len(subListIDCasesTest))
    for case_id in subListIDCasesTest:

        #Exam Image
        pathImageExam   = pathDataset + case_id + '/' + typeImages[0] + '/'
        #pathImageExam   = pathDataset +  case + '/' + typeImages[0] + '/'
        listCaseTestExam.append(pathImageExam)

        #Kidney Mask And Tumor Mask
        pathImageMasks   = pathDataset + case_id + '/' + typeImages[1] + '/'
        #pathImageMasks   = pathDataset +  case + '/' + typeImages[1] + '/'
        listCaseTestMasks.append(pathImageMasks)

    ########################################### Validation #################################################
    listCaseValidExam = []
    listCaseValidMasks = []

    sizeValidation = int((validationPercents/100) * sizeDataset)


    subListIDCasesValidation = []
    print("Current Size List Case: ",len(listIDCases))
    print("Size Validation: ",sizeValidation)

    ### Random Dataset
    if randomDataset:
        for i in range(sizeValidation):
            case = random.choice(listIDCases)
            subListIDCasesValidation.append(case)
            listIDCases.remove(case)
    ### Without Random Pacients
    else:
        for i in range(sizeValidation):
            case = listIDCases[0]
            subListIDCasesValidation.append(case)
            listIDCases.remove(case)

    print(len(listIDCases))
    for case in subListIDCasesValidation:

        #Kidney Image
        pathImageKidney   = pathDataset + case_id + '/' + typeImages[0] + '/'
        #pathImageKidney   = pathDataset +  case + '/'  + typeImages[0] + '/'
        listCaseValidExam.append(pathImageKidney)

        #Kidney Mask Add Tumor Mask
        pathImageMasks   = pathDataset + case_id + '/' + typeImages[1] + '/'
        #pathImageMasks   = pathDataset +  case + '/'  + typeImages[1] + '/'
        listCaseValidMasks.append(pathImageMasks)

    x_train_dir = listCaseTrainExam
    y_train_dir = listCaseTrainMasks
    x_test_dir = listCaseTestExam
    y_test_dir = listCaseTestMasks
    x_valid_dir = listCaseValidExam
    y_valid_dir = listCaseValidMasks
    return x_train_dir,y_train_dir,x_test_dir,y_test_dir,x_valid_dir,y_valid_dir