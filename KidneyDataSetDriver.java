import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;


public class KidneyDataSetDriver {
	
	static String INPUT_DATA_PATH =""; 
	static final String PRINT_FLAG ="PRINT"; 
	static final String NO_PRINT_FLAG ="NO_PRINT"; 
   	/**
	 * @param args
   	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		
		
		Path userPath = null;
		userPath = Paths.get(args[0]);
		if (!Files.exists(userPath) || !Files.isReadable(userPath)) {
			System.out.println("No file found at location, or cannot be read!! Exiting program!");
			System.exit(1);
		}
		
		System.out.println("Welcome to the analysis - Classification of Chronic Kidney Disease ");
		System.out.println("The flow of this analysis is iterative. Please follow the instructions at each step");
		System.out.println("Program will ask for various inputs at decisive steps and the respective results will be displayed");
		System.out.println("In case of no user input, press enter. Program will proceed with a default value ");
		System.out.println(" ");
		
		int iteration_count=0;
		String current_userQuery = null;
		BufferedReader buffread = null;
		
		for(;;){
		
		iteration_count++;
		System.out.println(" ");	
		System.out.println("Willkommen, we are in iteration #"+iteration_count);
		System.out.println("Step1: Summary of Raw Data will now be displayed......!!");
		System.out.println("Press 'R' to continue");
		buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
		current_userQuery = buffread.readLine(); 
		KidneyDataSetUtils utils = new KidneyDataSetUtils(args[0]); // Call this constructor to initialize the Instances and Source objects
	
		//Analysis
		
		
		//Step 1: Analysis - Show the Raw Data Stats
		if(current_userQuery.toUpperCase().trim().equals("R"))
			utils.loadAndPrintRawData(args[0],PRINT_FLAG);
		else
		{
			utils.loadAndPrintRawData(args[0],NO_PRINT_FLAG);	
			System.out.println("Data Set is loaded");
			System.out.println(" ");	
		}
		
		System.out.println(" ");
		System.out.println("Step2: Central Tendencies and Spread of Data will now be displayed......!!");
		System.out.println("Press 'C' to continue");
		buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
		current_userQuery = buffread.readLine(); 
		
		//Step 2: Analysis - Central tendency and variability in data
		if(current_userQuery.toUpperCase().trim().equals("C"))
		utils.centralAndVariationAnalysis(PRINT_FLAG,utils.getDataset());
		else
		{
		utils.centralAndVariationAnalysis(NO_PRINT_FLAG,utils.getDataset());	
		System.out.println("Central tendencies and spread have been calculated");
		System.out.println(" ");	
		}
			
		//Step 3: proceeding to missing value treatment
		System.out.println("Step3: Missing value Treatment..");
		System.out.println("Total Instances ="+utils.getDataset_size());
		System.out.println("# of instances without any missing values "+utils.getPure_instances());
		System.out.println(Math.ceil(100*(utils.getPure_instances()*100/utils.dataset_size))/100+" % of isntances are without missing value");
		System.out.println("Press 'M' to continue");
		buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
		current_userQuery = buffread.readLine();
		
		if(current_userQuery.toUpperCase().trim().equals("M"))
			utils.missingValueReplacement(PRINT_FLAG);
		else
		{
		utils.missingValueReplacement(NO_PRINT_FLAG);
		System.out.println("Missing values have been replaced");
		System.out.println(" ");	
		}
		
		
		//Step 4: proceeding to standardization and normalization
		System.out.println(" ");
		System.out.println("Step4: Normalization & Standardization..");
		System.out.println("Press 'NS' to continue");
		buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
		current_userQuery = buffread.readLine();
		
		if(current_userQuery.toUpperCase().trim().equals("NS"))
			utils.normalizeStndizeData(PRINT_FLAG);
		else
		{
		utils.normalizeStndizeData(NO_PRINT_FLAG);
		System.out.println("Data is normalized and standardized");
		System.out.println(" ");	
		}
		
		
		
		
		//Step 5: proceeding to attribute selection treatment
		System.out.println("Step5: Attribute Subset/Dimensionality Reduction..");
		System.out.println("Attributes in original dataset = "+utils.getOriginal_num_of_attrib());
		System.out.println("Press 'A' to continue");
		buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
		current_userQuery = buffread.readLine();
		
		if(current_userQuery.toUpperCase().trim().equals("A"))
			utils.dimensionReduction(PRINT_FLAG);
		else
		utils.dimensionReduction(NO_PRINT_FLAG);
		
		System.out.println("Attributes have been reduced to "+utils.getReduced_num_of_attrib());
		System.out.println(" ");
		
		
		
		System.out.println("****************** Revisiting - Data after completing preprocessing steps***************");
		System.out.println("Press 'RV' to continue");
		buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
		current_userQuery = buffread.readLine();
		if(current_userQuery.toUpperCase().trim().equals("RV"))
		{
			
			System.out.println(utils.getDataset_with_replmnts().toSummaryString());
			
		}

		
		
		
		//Step 6: proceeding to Classification
		System.out.println("Step6: Classification by NaiveBayes with Cross Validation");
		System.out.println("Press 'NB' to continue");
		buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
		current_userQuery = buffread.readLine();
		
		if(current_userQuery.toUpperCase().trim().equals("NB")){
			
			for(;;){
			System.out.println("Proceeding for Naive Bayes Classification, Enter a K-value (>2) for folding in cross validation...");
			buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
			current_userQuery = buffread.readLine();
			if(current_userQuery.matches("\\d+")){ //if number is entered by user
			if(Integer.parseInt(current_userQuery)>2 && Integer.parseInt(current_userQuery)<utils.getDataset_size())
			utils.naiveBayesClassification(Integer.parseInt(current_userQuery));
			}
			else
				utils.naiveBayesClassification(5);//default value for 
			
			System.out.println("Do you want to repeat Naive Bayes again with another K-value? Enter - Y/N");
			buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
			current_userQuery = buffread.readLine();
			
			if(current_userQuery.toUpperCase().trim().equals("N"))
				break;
			else
				continue;
			
			}
		}else
			utils.naiveBayesClassification(5);//run for default folds 
		
		//Step 7: Second Classification
		System.out.println("Step7: Classification by Decision Tree(J48-post pruning) with Cross Validation");
		System.out.println("Press 'DT' to continue");
		buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
		current_userQuery = buffread.readLine();
		
		if(current_userQuery.toUpperCase().trim().equals("DT")){
			
		
		for(;;){
			System.out.println("Step8: Proceeding for J48, Enter a K-value(>2) for folding in cross validation...");
			buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
			current_userQuery = buffread.readLine();
			if(current_userQuery.matches("\\d+")){ //if number is entered by user, less than dataset size, else cross validation thro excep
			if(Integer.parseInt(current_userQuery)>2 && Integer.parseInt(current_userQuery)<utils.getDataset_size())
			utils.decisionTreeJ48Classifnc(Integer.parseInt(current_userQuery));
			}
			else
				utils.decisionTreeJ48Classifnc(5);//default value for 
			
			System.out.println("Do you want to repeat J48 again with another K-value? Enter - Y/N");
			buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
			current_userQuery = buffread.readLine();
			
			if(current_userQuery.toUpperCase().trim().equals("N"))
				break;
			else
				continue;
			
			}
		}
		else
			utils.decisionTreeJ48Classifnc(5);// run for default folds
		
		//Step 8: print to - whether user asks or doesnt ask
		int k = utils.printAndCompareModels();
		
		//Step 9: print to - whether user asks or doesnt ask
		System.out.println("Step9: Proceeding for Classification, with outlier removal as a pre-step...");
		System.out.println("Press-  Y/N to proceed");
		buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
		current_userQuery = buffread.readLine();
		if(current_userQuery.toUpperCase().trim().equals("Y")){ //if number is entered by user, less than dataset size, else cross validation thro excep
		  utils.classifcwithOutlierRemoval();
		}
		
		//Step 10 : Classification with Dynamic 
		System.out.println("Step10: Proceeding for Classification with iterative attribute reduction using Information Gain,...");
		System.out.println("Press 'AR' to continue");
		buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
		current_userQuery = buffread.readLine();
		if(current_userQuery.toUpperCase().trim().equals("AR"))
		{
			for(;;)
			{
		System.out.println("Please enter a threshold value between 0.1 to 0.45. If not sure, press enter, a default 0.2 will be chosen");
		KidneyDataSetUtils util1=new KidneyDataSetUtils(args[0]);
		buffread = new BufferedReader(new InputStreamReader(System.in));
		current_userQuery = buffread.readLine();
		boolean b3 = Pattern.matches("([0-9]*)\\.([0-9]*)", current_userQuery);		
		if(b3){ //if number is entered by user, less than dataset size, else cross validation thro excep
			  utils.classifcWithDynamicAttribs(Double.parseDouble(current_userQuery),util1.getDataset());
			  
			}
			else
				utils.classifcWithDynamicAttribs((double)0.2,util1.getDataset())
		
				System.out.println("Do you want to repeat with a different threshold value? Enter - Y/N");
				buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console 
				current_userQuery = buffread.readLine();
				if(current_userQuery.toUpperCase().trim().equals("N"))
					break;
				else
					continue;
			}
		}
		
		
		
		//Quitting Block
		System.out.println("All Steps Done - about to quit to program..We are at iteration Number "+iteration_count);
		System.out.println("Press 'Q' to quit; Any other key to start next iteration of analysis");
		buffread = new BufferedReader(new InputStreamReader(System.in));// capture user input from console for searching
		current_userQuery = buffread.readLine();
		if(current_userQuery.toUpperCase().equals("Q")){
			//System.out.println("Auf Wiedersehen, we are quitting program at iteration #");
			break;
				
		}
		
		
		}
		
		if(buffread!=null)
		buffread.close();
		
		
		

	}

}
