import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.classifiers.trees.J48;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.EnumHelper;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.matrix.Maths;
import weka.experiment.Stats;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.instance.RemoveMisclassified;


public class KidneyDataSetUtils {
	static final String PRINT_FLAG ="PRINT"; 
	static final String NO_PRINT_FLAG ="NO_PRINT"; 
	static Map<String, String> compare_models = new HashMap();
	static double[] nb_result_arr = null;
	static double[] dt_result_arr = null;

	public KidneyDataSetUtils(String path) throws Exception {
		// TODO Auto-generated constructor stub
		//This constructor will create new objects and initialize the source and instances
		this.source = new DataSource(path); // Weka - DataSource
		this.dataset = source.getDataSet(); // Weka - Instances
		this.dataset_with_replmnts = source.getDataSet();
		
	}
	
	private static final String FILE_WRITE_LOCATION = "/FileWritebyProgram";
	private static final String FILE_WRITE_NAME="/kidneyAllInput.csv";
	private static final String DISPLAY= "***************";
	private static final int List = 0;
	
	
	int original_num_of_attrib=0;
	int original_num_of_instances=0;
	int pure_instances=0; //num of instance w/o any missing values
	
	public int getPure_instances() {
		return pure_instances;
	}

	public void setPure_instances(int pure_instances) {
		this.pure_instances = pure_instances;
	}

	int reduced_num_of_attrib=0;
	public int getNo_of_class_labels() {
		return no_of_class_labels;
	}

	public void setNo_of_class_labels(int no_of_class_labels) {
		this.no_of_class_labels = no_of_class_labels;
	}

	int no_of_class_labels=0;
	int reduced_num_of_instances=0;
	
	float avg_missing_values_of_all_attribs=0;
	int dataset_size=0;
	
	DataSource source =null;
	Instances dataset =null;
	Instances dataset_with_replmnts =null;
	
	
	public Instances getDataset_with_replmnts() {
		return dataset_with_replmnts;
	}

	public void setDataset_with_replmnts(Instances dataset_with_replmnts) {
		this.dataset_with_replmnts = dataset_with_replmnts;
	}

	public DataSource getSource() {
		return source;
	}

	public void setSource(DataSource source) {
		this.source = source;
	}

	public Instances getDataset() {
		return dataset;
	}

	public void setDataset(Instances dataset) {
		this.dataset = dataset;
	}

	public void loadAndPrintRawData(String path, String print_flag) throws Exception{
		Path userPath = Paths.get(path);
		Path userParent = userPath.getParent();
		Path p1 = Paths.get(userParent + FILE_WRITE_LOCATION);
		
		this.dataset.setClassIndex(dataset.numAttributes()-1);
		
		if(print_flag.equals(PRINT_FLAG)){
		System.out.println("");
		System.out.println(DISPLAY+"Raw Data Starts"+DISPLAY);
		System.out.println(dataset.toSummaryString());
		System.out.println(DISPLAY+"Raw Data Ends"+DISPLAY);
		System.out.println("");
		}
		
		
		///Set these global values for use - with of without print flag
		setOriginal_num_of_attrib(dataset.numAttributes());
		setDataset_size(dataset.size());
		setOriginal_num_of_instances(dataset.numInstances());
		setDataset(dataset);// this object can now be accessed throughout!
		
		
	}
	
	public void centralAndVariationAnalysis(String print_flag, Instances dataset1) throws Exception{
		
		if(print_flag.equals(PRINT_FLAG))
		{
		System.out.println(" ");
	    System.out.println(DISPLAY+"Central Tendencies and Variation - Starts"+DISPLAY);
	    System.out.println("Normal distribution is checked by approximation ~ 68-95-99.7% rule");
	    System.out.println(" ");
		}
		dataset1.setClassIndex(dataset1.numAttributes()-1);	    
		
		for (int i = 0; i < dataset1.numAttributes(); i++) {
			//Check the correlation of a given attribute with class labels
			double correlation_val=0;
			
			CorrelationAttributeEval corr = new CorrelationAttributeEval();
			
			corr.buildEvaluator(dataset1);
			correlation_val = corr.evaluateAttribute(i);
			correlation_val = Math.ceil(correlation_val*1000)/1000;
					
			if (dataset1.attribute(i).isNumeric()) {
				double max =  Math.ceil(1000*dataset1.attributeStats(i).numericStats.max)/1000;
				double min =  Math.ceil(1000*dataset1.attributeStats(i).numericStats.min)/1000;
				double mean =  Math.ceil(1000*dataset1.attributeStats(i).numericStats.mean)/1000;
				double sd =  Math.ceil(1000*dataset1.attributeStats(i).numericStats.stdDev)/1000;
				
				double range =  Math.ceil(1000*max-min)/1000;
				
				boolean normal_dist = false;//checkDitri();
				
				   normal_dist = checkifApproxNormalDistrib(dataset1.attributeToDoubleArray(i),mean,sd,dataset1.attribute(i).toString());
				   double median = Math.ceil(1000*Utils.kthSmallestValue(dataset1.attributeToDoubleArray(i), dataset1.attributeToDoubleArray(i).length/2))/1000;
				   //median<mean ==> right skewed
				   //median?mean ==> left skewed
				   
				  
				if(print_flag.equals(PRINT_FLAG))
				{
					double skew=0;
					skew = mean-median;
				  System.out.println("Attribute = "+dataset1.attribute(i).name());
				  System.out.println(" | Mean = "+mean+" | Median ="+median+" | Stnd Dev = "+sd+" | Correlation with Class = "+correlation_val+" | Range ="+range+" Max ="+max+" Min ="+min);
				  if(mean>median && normal_dist==false)
				  System.out.println(" Distribution is somewhat right skewed");
				  else if(mean<median && normal_dist==false)
				  System.out.println(" Distribution is somewhat left skewed");
				  else if(normal_dist==true)
				  System.out.println(" Distribution is approximately bell curve (normal)");
				  
				  System.out.println(" ");
				}
			}
			else if(dataset1.attribute(i).isNominal()){
				//for nominal attributes attributes, print the mode
				dataset1.setClassIndex(dataset1.numAttributes()-1);
				if(correlation_val>0){
					if(print_flag.equals(PRINT_FLAG))
					{
					System.out.println("For Nominal Attributes");
					System.out.println("Attribute = "+dataset.attribute(i).name().toString()+" | Correlation with Class = "+correlation_val);
					System.out.println("");  
					}
				}
			}
			
		}
		
		if(print_flag.equals(PRINT_FLAG))
		{
		System.out.println(DISPLAY+"Central Tendencies and Variation - Ends"+DISPLAY);
		System.out.println(" ");
		}
		
		//these are to be set
		int pure_instance_count=0;
		int missing_element_count=0;
		for(int j=0;j<dataset1.size();j++){
			if(!dataset1.instance(j).toString().contains("?"))
				pure_instance_count++;
			
		}
		
		setPure_instances(pure_instance_count);
	}
	
	private boolean checkifApproxNormalDistrib(double[] column_vector, double mean, double sd,String atrib){
		//Note that this method will run for a given attribute
		
		boolean normalDis = false;
		double meu_minus_Sigma=0;
		double meu_plus_Sigma=0;
		meu_minus_Sigma = mean-sd;
		meu_plus_Sigma = mean+sd;
		int records_needed_in_2_sigma =0; //~ 68%
		int records_present_in_2_sigma =0;
		
		double meu_minus_two_Sigma=0;
		double meu_plus_two_Sigma=0;
		meu_minus_two_Sigma = mean - (2*sd);
		meu_plus_two_Sigma = mean+(2*sd);
		int records_needed_in_4_sigma =0; //~95%
		int records_present_in_4_sigma =0;
		
		double meu_minus_6_Sigma=0;
		double meu_plus_6_Sigma=0;
		meu_minus_6_Sigma = mean - (6*sd);
		meu_plus_6_Sigma = mean+(6*sd);
		int records_needed_in_6_sigma =0; //~99.7%
		int records_present_in_6_sigma =0;
		
		
		
		//2 sigma and 4 sigma check - for a given attribute
		for(int i=0;i<column_vector.length;i++){
			
			if((column_vector[i]) >=meu_minus_Sigma || column_vector[i]<=meu_plus_Sigma)
				records_present_in_2_sigma++;
			
			if(column_vector[i]>=meu_minus_two_Sigma || column_vector[i]<=meu_plus_two_Sigma)
				records_present_in_4_sigma++;
			
			if(column_vector[i]>=meu_minus_6_Sigma || column_vector[i]<=meu_plus_6_Sigma)
				records_present_in_6_sigma++;
			
		}
		
		records_needed_in_2_sigma = (int) (column_vector.length*0.68);
		records_needed_in_4_sigma = (int) (column_vector.length*0.95);
		records_needed_in_6_sigma = (int) (column_vector.length*0.997);
		
		if(Math.abs(records_needed_in_2_sigma-records_present_in_2_sigma) <= 5 
				|| Math.abs(records_needed_in_4_sigma-records_present_in_4_sigma)<=5
				|| Math.abs(records_needed_in_6_sigma-records_present_in_6_sigma)<=5 ){
			normalDis=true;
			}
		
		return normalDis;
	}
	
	
	public void missingValueReplacement(String print_flag) throws Exception{
		if(print_flag.equals(PRINT_FLAG))
		{
		System.out.println("");
		System.out.println(DISPLAY+"Replace Missing Values start"+DISPLAY);
		}
		
		ReplaceMissingValues repl= new ReplaceMissingValues();
		repl.setInputFormat(this.dataset_with_replmnts);
		dataset_with_replmnts=Filter.useFilter(dataset_with_replmnts, repl);
		setDataset_with_replmnts(dataset_with_replmnts);// set this object, for use in other methods
		
		if(print_flag.equals(PRINT_FLAG))
		{
		System.out.println("");
		System.out.println("Dataset after mssing value replacment ="+dataset_with_replmnts.toSummaryString());
		System.out.println(DISPLAY+"Replace Missing Values ends"+DISPLAY);
		System.out.println("");
		}
		
		System.out.println("");
	}
	
	public void normalizeStndizeData(String print_flag) throws Exception{
		if(print_flag.equals(PRINT_FLAG))
		{
		System.out.println("");
		System.out.println(DISPLAY+"Normalization & Standardization start"+DISPLAY);
		}
		Normalize norm = new Normalize();
		norm.setInputFormat(getDataset_with_replmnts());
		
		if(print_flag.equals(PRINT_FLAG))
		{
		System.out.println("Normalization Result is "+getDataset_with_replmnts().toSummaryString());
		}
		
		Standardize stnd = new Standardize();
		stnd.setInputFormat(getDataset_with_replmnts());
		
		
		if(print_flag.equals(PRINT_FLAG))
		{
		System.out.println("");
		System.out.println(DISPLAY+"Normalization & Standardization ends"+DISPLAY);
		System.out.println("");
		}
		
		
	}
	
	public void dimensionReduction(String print_flag) throws Exception{
		if(print_flag.equals(PRINT_FLAG))
		{
		System.out.println("");
		System.out.println(DISPLAY+"Attribute Reduction  start"+DISPLAY);
		}
		
		AttributeSelection filter = new AttributeSelection(); // create and initiate a new AttributeSelection instance
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(this.dataset_with_replmnts);
		
		Instances newData = Filter.useFilter(dataset_with_replmnts, filter);
		this.dataset_with_replmnts = newData;
		setDataset_with_replmnts(newData);
		
		if(print_flag.equals(PRINT_FLAG))
		{
		System.out.println(newData.toSummaryString());
		System.out.println(DISPLAY+"Attribute Reduction end"+DISPLAY);
		System.out.println("");
		}
		setReduced_num_of_attrib(newData.numAttributes());
	}
	
	public void naiveBayesClassification(int k_val) throws Exception{
		System.out.println("");
		System.out.println(DISPLAY+"Classification by NaiveBayes with "+k_val+" Folds"+DISPLAY);
		
		DataSource s1 = getSource();
		
		
		//set class index to the last attribute
		dataset_with_replmnts.setClassIndex(dataset_with_replmnts.numAttributes()-1);
		
		ReplaceMissingValues repl= new ReplaceMissingValues();
		repl.setInputFormat(this.dataset_with_replmnts);
		dataset_with_replmnts=Filter.useFilter(dataset_with_replmnts, repl);
		Normalize norm = new Normalize();
		norm.setInputFormat(dataset_with_replmnts);
		Standardize stnd = new Standardize();
		stnd.setInputFormat(dataset_with_replmnts);
		AttributeSelection filter = new AttributeSelection(); // create and initiate a new AttributeSelection instance
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(dataset_with_replmnts);
		
		Instances newData = Filter.useFilter(dataset_with_replmnts, filter);
		dataset_with_replmnts = newData;
		
		dataset_with_replmnts.setClassIndex(dataset_with_replmnts.numAttributes()-1);
	
		NaiveBayes nb = new NaiveBayes();//NaiveBayesMultinomialText
		nb.buildClassifier(dataset_with_replmnts);
		
		Evaluation eval1 = new Evaluation(dataset_with_replmnts);
		Random rand = new Random(1);
		int folds = k_val;
		
		
		eval1.crossValidateModel(nb, dataset_with_replmnts, folds, rand);
		System.out.println(eval1.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("Correct % = "+eval1.pctCorrect());
		System.out.println("Incorrect % = "+eval1.pctIncorrect());
		System.out.println("Precision = "+eval1.precision(1));
		System.out.println("Recall = "+eval1.recall(1));
		System.out.println("Error Rate = "+eval1.errorRate());
	    //the confusion matrix
		System.out.println(eval1.toMatrixString("=== Overall Confusion Matrix ===\n"));
		
		nb_result_arr = new double [7];
		nb_result_arr[0] = Math.ceil(1000*eval1.pctCorrect())/1000;
		nb_result_arr[1] = Math.ceil(1000*eval1.pctIncorrect())/1000;
		nb_result_arr[2] = Math.ceil(1000*eval1.precision(1))/1000;
		nb_result_arr[3] = Math.ceil(1000*eval1.recall(1)/1000);
		nb_result_arr[4] = Math.ceil(1000*eval1.errorRate())/1000;
		nb_result_arr[5] = Math.ceil(1000*eval1.areaUnderROC(1))/1000;
		nb_result_arr[6] = Math.ceil(1000*eval1.rootMeanSquaredError())/1000;
		
		System.out.println(DISPLAY+"Classification by NaiveBaye end"+DISPLAY);
		System.out.println("");
	}
	
	public <T> void naiveBayesClassificationWithSampling(int k_val, float sampl_ratio) throws Exception{
		System.out.println("");
		System.out.println(DISPLAY+"Classification by NaiveBayes with random sampling by ratio = "+sampl_ratio+" starts.."+DISPLAY);
		
		DataSource s1 = getSource();
		
		
		//set class index to the last attribute
		dataset_with_replmnts.setClassIndex(dataset_with_replmnts.numAttributes()-1);
		
		ReplaceMissingValues repl= new ReplaceMissingValues();
		repl.setInputFormat(this.dataset_with_replmnts);
		dataset_with_replmnts=Filter.useFilter(dataset_with_replmnts, repl);
		Normalize norm = new Normalize();
		norm.setInputFormat(dataset_with_replmnts);
		Standardize stnd = new Standardize();
		stnd.setInputFormat(dataset_with_replmnts);
		AttributeSelection filter = new AttributeSelection(); // create and initiate a new AttributeSelection instance
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(dataset_with_replmnts);
		
		Instances newData = Filter.useFilter(dataset_with_replmnts, filter);
		dataset_with_replmnts = newData;
		
		dataset_with_replmnts.setClassIndex(dataset_with_replmnts.numAttributes()-1);
		
		
		//Step - create a Map out of the instances - to be sent for sampling
		
		Map<Integer, String> input_map = new TreeMap<Integer, String>();
		for(int r=0;r<dataset_with_replmnts.size();r++){
			input_map.put(r+1, dataset_with_replmnts.instance(r).toString());
		}
		DataSampling dataSmp = new DataSampling();
		dataSmp.setInput_data_map(input_map);
		dataSmp.sampleData("Random", sampl_ratio);
		
		Map<Integer, String> train_map = new TreeMap<Integer, String>();
		Map<Integer, String> test_map = new TreeMap<Integer, String>();
		
		train_map = dataSmp.getTraining_data_Sampled();
		test_map = dataSmp.getTesting_data_Sampled();
		
		List<String> train_list = new ArrayList<String>(train_map.values());
		List<String> test = new ArrayList<String>(test_map.values());
		
		System.out.println("train data size = "+train_map.size()+" test data size = "+test_map.size());
		
		dataset_with_replmnts = getInstancesFromMap(train_map,dataset_with_replmnts);
		
		NaiveBayes nb = new NaiveBayes();//NaiveBayesMultinomialText
		nb.buildClassifier(dataset_with_replmnts);
		
		Evaluation eval1 = new Evaluation(dataset_with_replmnts);
		Random rand = new Random(1);
		int folds = k_val;
		
		//now add test data, before evaluation
		dataset_with_replmnts = getInstancesFromMap(test_map,dataset_with_replmnts);
		
		eval1.crossValidateModel(nb, dataset_with_replmnts, folds, rand);
		System.out.println(eval1.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("Correct % = "+eval1.pctCorrect());
		System.out.println("Incorrect % = "+eval1.pctIncorrect());
		System.out.println("Precision = "+eval1.precision(1));
		System.out.println("Recall = "+eval1.recall(1));
		System.out.println("Error Rate = "+eval1.errorRate());
	    //the confusion matrix
		System.out.println(eval1.toMatrixString("=== Overall Confusion Matrix ===\n"));
		
		nb_result_arr = new double [7];
		nb_result_arr[0] = Math.ceil(1000*eval1.pctCorrect())/1000;
		nb_result_arr[1] = Math.ceil(1000*eval1.pctIncorrect())/1000;
		nb_result_arr[2] = Math.ceil(1000*eval1.precision(1))/1000;
		nb_result_arr[3] = Math.ceil(1000*eval1.recall(1)/1000);
		nb_result_arr[4] = Math.ceil(1000*eval1.errorRate())/1000;
		nb_result_arr[5] = Math.ceil(1000*eval1.areaUnderROC(1))/1000;
		nb_result_arr[6] = Math.ceil(1000*eval1.rootMeanSquaredError())/1000;
		
		System.out.println(DISPLAY+"Classification by NaiveBaye with random sampling by ratio = "+sampl_ratio+" ends"+DISPLAY);
		System.out.println("");
	}

	private Instances getInstancesFromMap(Map<Integer, String> test_map, Instances ds1) {
		Instance inst = new DenseInstance(test_map.size()); 
		
		for(String val:test_map.values() ){
			String [] temp = val.split(",");
			inst = new DenseInstance(25); 
			 	for(int i=0;i<temp.length;i++){	 
				
				inst.setValue(i, temp[i]);
				
			 	}
			 	ds1.add(inst);
			
		}
		return ds1;
	}
	
	public void decisionTreeJ48Classifnc(int fold_val) throws Exception{
		System.out.println("");
		System.out.println(DISPLAY+"Classification by Decision Tree-J48 with "+fold_val+" Fold Evaluation "+DISPLAY);
		
		//set class index to the last attribute
		dataset_with_replmnts.setClassIndex(dataset_with_replmnts.numAttributes()-1);
	
		J48 j48 = new J48();
		j48.buildClassifier(dataset_with_replmnts);
		
		Evaluation eval1 = new Evaluation(dataset_with_replmnts);
		Random rand = new Random(1);
		int folds = fold_val;
		
		eval1.crossValidateModel(j48, dataset_with_replmnts, folds, rand);
		System.out.println(eval1.toSummaryString("Evaluation results:\n", false));
		System.out.println("Correct % = "+eval1.pctCorrect());
		System.out.println("Incorrect % = "+eval1.pctIncorrect());
		System.out.println("Precision = "+eval1.precision(1));
		System.out.println("Recall = "+eval1.recall(1));
		System.out.println("Error Rate = "+eval1.errorRate());
	    //the confusion matrix
		System.out.println(eval1.toMatrixString("=== Overall Confusion Matrix ===\n"));
		
		System.out.println("");
		System.out.println("Graph........");
		System.out.println(j48.graph());
		
		dt_result_arr = new double [7];
		dt_result_arr[0] = Math.ceil(1000*eval1.pctCorrect())/1000;
		dt_result_arr[1] = Math.ceil(1000*eval1.pctIncorrect())/1000;
		dt_result_arr[2] = Math.ceil(1000*eval1.precision(1))/1000;
		dt_result_arr[3] = Math.ceil(1000*eval1.recall(1))/1000;
		dt_result_arr[4] = Math.ceil(1000*eval1.errorRate())/1000;
		dt_result_arr[5] = Math.ceil(1000*eval1.areaUnderROC(1))/1000;
		dt_result_arr[6] = Math.ceil(1000*eval1.rootMeanSquaredError())/1000;
		
		System.out.println(DISPLAY+"Classification by Decision Tree-J48 with "+fold_val+" Folds Evaluation end"+DISPLAY);
		System.out.println("");
	}
	
	
	
	public int printAndCompareModels() {
		
		
		if(nb_result_arr==null|| dt_result_arr==null)
			return -1;
		
		if(nb_result_arr.length==0 || dt_result_arr.length==0)
			return -2;
		
		if(nb_result_arr.length==7 && dt_result_arr.length==7)
		{
		System.out.println("");
		System.out.println(DISPLAY+" Comparing Models Starts"+DISPLAY);
		System.out.println("");
			System.out.println("Param     NaiBays     DecTree(J48)");// 10 5
		
			System.out.println("%Correc = "+nb_result_arr[0]+"          "+dt_result_arr[0]);
			System.out.println("%InCorr = "+nb_result_arr[1]+"          "+dt_result_arr[1]);
			System.out.println("Precisi = "+nb_result_arr[2]+"          "+dt_result_arr[2]);
			System.out.println("Recall  = "+nb_result_arr[3]+"          "+dt_result_arr[3]);
			System.out.println("ErrRate = "+nb_result_arr[4]+"          "+dt_result_arr[4]);
			System.out.println("AreaRoC = "+nb_result_arr[5]+"          "+dt_result_arr[5]);
			System.out.println("RMSE    = "+nb_result_arr[6]+"          "+dt_result_arr[6]);
		
		
		System.out.println("");
		System.out.println(DISPLAY+" Comparing Models Ends"+DISPLAY);
		System.out.println("");
		}
		return 1;
		
	}
	

	public int getOriginal_num_of_instances() {
		return original_num_of_instances;
	}




	public void setOriginal_num_of_instances(int original_num_of_instances) {
		this.original_num_of_instances = original_num_of_instances;
	}




	public int getReduced_num_of_instances() {
		return reduced_num_of_instances;
	}




	public void setReduced_num_of_instances(int reduced_num_of_instances) {
		this.reduced_num_of_instances = reduced_num_of_instances;
	}




	public int getOriginal_num_of_attrib() {
		return original_num_of_attrib;
	}

	public void setOriginal_num_of_attrib(int original_num_of_attrib) {
		this.original_num_of_attrib = original_num_of_attrib;
	}

	public int getReduced_num_of_attrib() {
		return reduced_num_of_attrib;
	}

	public void setReduced_num_of_attrib(int reduced_num_of_attrib) {
		this.reduced_num_of_attrib = reduced_num_of_attrib;
	}

	public float getAvg_missing_values_of_all_attribs() {
		return avg_missing_values_of_all_attribs;
	}

	public void setAvg_missing_values_of_all_attribs(
			float avg_missing_values_of_all_attribs) {
		this.avg_missing_values_of_all_attribs = avg_missing_values_of_all_attribs;
	}

	public int getDataset_size() {
		return dataset_size;
	}

	public void setDataset_size(int dataset_size) {
		this.dataset_size = dataset_size;
	}

	public void classifcwithOutlierRemoval() throws Exception {
		System.out.println("");
		System.out.println(DISPLAY+"Outlier Removal followed by Naive Bayes Classification starts"+DISPLAY);
		
		DataSource s1 = getSource();
		
		//set class index to the last attribute
		this.dataset.setClassIndex(dataset.numAttributes()-1);
		
		
		ReplaceMissingValues repl= new ReplaceMissingValues();
		repl.setInputFormat(this.dataset);
		dataset=Filter.useFilter(dataset, repl);
		
		
		Normalize norm = new Normalize();
		norm.setInputFormat(dataset);
		Standardize stnd = new Standardize();
		stnd.setInputFormat(dataset);
		
		AttributeSelection filter = new AttributeSelection(); // create and initiate a new AttributeSelection instance
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(dataset);
		
		Instances newData = Filter.useFilter(dataset, filter);
		dataset = newData; 
		
		dataset.setClassIndex(dataset.numAttributes()-1);
	
		RemoveMisclassified filter1 = new RemoveMisclassified();
		filter1.setClassIndex(-1);
		filter1.setNumFolds(0);
		filter1.setThreshold(0.1);
		filter1.setMaxIterations(0);
		filter1.setInputFormat(dataset);
		dataset = Filter.useFilter(dataset, filter1);
		System.out.println("*** Original Number of instances ***"+dataset_size);
		System.out.println("*** Number of Instances after outlier detection is ***"+dataset.size());
	    
		NaiveBayes nb = new NaiveBayes();//NaiveBayesMultinomialText
		nb.buildClassifier(dataset);
		
		Evaluation eval1 = new Evaluation(dataset);
		Random rand = new Random(1);
		int folds = 5;
		
		
		eval1.crossValidateModel(nb, dataset_with_replmnts, folds, rand);
		System.out.println(eval1.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("Correct % = "+eval1.pctCorrect());
		System.out.println("Incorrect % = "+eval1.pctIncorrect());
		System.out.println("Precision = "+eval1.precision(1));
		System.out.println("Recall = "+eval1.recall(1));
		System.out.println("Error Rate = "+eval1.errorRate());
	    //the confusion matrix
		System.out.println(eval1.toMatrixString("=== Overall Confusion Matrix ===\n"));
		
		System.out.println(DISPLAY+"Outlier Removal followed by Naive Bayes Classification ends"+DISPLAY);
		System.out.println("");
		
	}

	public void classifcWithDynamicAttribs(float info_gain, Instances its) throws Exception {
		System.out.println("");
		System.out.println(DISPLAY+"Classification with Dynamic Attribute Reduction with threshold ="+info_gain+",starts"+DISPLAY);
		
		AttributeSelection filter = new AttributeSelection(); // create and initiate a new AttributeSelection instance
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker search = new Ranker();
	    search.setGenerateRanking(true);
	    search.getThreshold();
	    search.setThreshold(info_gain);
		filter.setEvaluator(eval);
		filter.setSearch(search);
		
		filter.setInputFormat(its);
		System.out.println("**Original attributes "+its.numAttributes());
	    Instances newData = Filter.useFilter(its, filter);
	    System.out.println("**Reduced attributes "+newData.numAttributes());
	    
	    System.out.println(newData.toSummaryString());
	    
	    newData.setClassIndex(newData.numAttributes()-1);
	    NaiveBayes nB=new NaiveBayes();
		nB.buildClassifier(newData);
		Evaluation eval1=new Evaluation(newData);
	    eval1.crossValidateModel(nB,newData,10,new Random(1));
		System.out.println(eval1.toSummaryString("\nResults\n=====\n",true));
		
		System.out.println(DISPLAY+"Classification with Dynamic Attribute Reduction ends"+DISPLAY);
		System.out.println("");
	}

	

	
	
	
	

}
