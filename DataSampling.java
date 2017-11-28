import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;




public class DataSampling {
	
	public static final float SAMPLING_RATIO = (float) 0.6667; //i.e 0.75 ~ 75% goes into Training data from whole sample
	//0.9421; ==> means that approx 100 test data will be selected out of 1728 car data set
	
	public static float getSamplingRatio() {
		return SAMPLING_RATIO;
	}

	static final String SAMPLING_METHOD ="Random"; // Set to Default
	
	static final String SAMPLING_METHOD_HOPP ="Odd/Even Hopp";//  "Hopp" - takes 75% in hopping way
	static final String SAMPLING_METHOD_BLOCK ="Block";//"Block" => takes 75% block from the beginning,
	static final String SAMPLING_METHOD_RANDOM ="Random";
	static final String METADATA_CAR_DATA_SET = "buying"+","+"maint"+","+"doors"+","+"persons"+","+"logboot"+","+"safety"+","+"Eval";
	static final String WRITE_TEST_DATA_FILE_LOCATION = "/home/sayantan/Desktop/cardata_tst1.csv";
	static final String WRITE_TRAIN_DATA_FILE_LOCATION = "/home/sayantan/Desktop/cardata_trn1.csv";
	private static final String COMMA_DELIMITER = ",";
	private static final String NEW_LINE_SEPARATOR = "\n";

	public DataSampling() {
		// TODO Auto-generated constructor stub
		testing_data_Sampled = new TreeMap<Integer, String>();
		training_data_Sampled = new TreeMap<Integer, String>();
		input_data_map = new TreeMap<Integer, String>();
		
	}
	
	Map<Integer, String>   input_data_map =null;
	public Map<Integer, String> getInput_data_map() {
		return input_data_map;
	}


	public void setInput_data_map(Map<Integer, String> input_data_map) {
		this.input_data_map = input_data_map;
	}


	public Map<Integer, String> getTraining_data_Sampled() {
		return training_data_Sampled;
	}


	public void setTraining_data_Sampled(Map<Integer, String> training_data_Sampled) {
		this.training_data_Sampled = training_data_Sampled;
	}


	public Map<Integer, String> getTesting_data_Sampled() {
		return testing_data_Sampled;
	}


	public void setTesting_data_Sampled(Map<Integer, String> testing_data_Sampled) {
		this.testing_data_Sampled = testing_data_Sampled;
	}

	Map<Integer, String>   training_data_Sampled =null;
	Map<Integer, String>   testing_data_Sampled =null;
	
	
public  void sampleData(String sampling_method, float sampling_ratio) throws IOException{
		
	if(sampling_method.equals(""))
		sampling_method = SAMPLING_METHOD_RANDOM;
	
		int training_data_Max=-1;
		int test_data_Max=-1;
		
		int training_data_count=0;
		int test_data_count=0;
		
		training_data_Max = (int) (input_data_map.size()*sampling_ratio);// ==> 1000*.75 = 750/751 // SAMPLING_RATIO
		test_data_Max = input_data_map.size()-training_data_Max;// ==> 1000 -750/751 = 250/249
			
		//SAMPLING_METHOD_HOPP
		if(SAMPLING_METHOD.equals(sampling_method)){ // Hopping by odd even numbers
		for(Map.Entry<Integer, String> m : input_data_map.entrySet()  ){
			
			
			if( (m.getKey().intValue())%2==0 || test_data_count==test_data_Max) { // i.e when we get even numbers OR 'leftovers', which are odd, put them in training set
				training_data_count++; // increase counter
				if(training_data_count<=training_data_Max)
					training_data_Sampled.put(m.getKey(), m.getValue()); //Training_Samples.put(m.getKey(), m.getValue());
				
			}
			else if (! (m.getKey()%2==0)) { // i.e get odd numbers, put them in test set
				test_data_count++;
						if(test_data_count<=test_data_Max) // 1,2,3... < 250
						testing_data_Sampled.put(m.getKey(), m.getValue()); //Test_Samples.put(m.getKey(), m.getValue());
					
				}
			
			}
		}
		else if(SAMPLING_METHOD.equals(sampling_method)){ //SAMPLING_METHOD_BLOCK
			for(Map.Entry<Integer, String> m : input_data_map.entrySet()  ){
								
				if(m.getKey().intValue()<=training_data_Max)
					training_data_Sampled.put(m.getKey().intValue(), m.getValue()); //Training_Samples.put(m.getKey().intValue(), m.getValue());
				else
					testing_data_Sampled.put(m.getKey().intValue(), m.getValue()); //Test_Samples.put(m.getKey().intValue(), m.getValue());
			}
			
		}
		else if(SAMPLING_METHOD.equals(sampling_method)){ //SAMPLING_METHOD_RANDOM
			Map<Integer, String> all_data = new TreeMap<Integer, String>();
			all_data = input_data_map; //copy the input data, input_data_map_clone
			//Step1: Select a random number between the total range of the data
			
			Random rnd = new Random();
			int min =1;
			int max =input_data_map.size();
			int range = max -min+1;
			boolean generateNumBool =false;
			int randomKey=-1;
			int loop=0;
			
			testing_data_Sampled.clear();
			training_data_Sampled.clear();
			
			while(generateNumBool==false )
			{
				randomKey = (rnd.nextInt(range)+min);
				loop++;
				
				Iterator itr  = all_data.entrySet().iterator();
				int mapKey=-1;
					while(itr.hasNext())
					{
						Map.Entry entry = (Map.Entry) itr.next();
						mapKey = (Integer) entry.getKey();
						
						if(mapKey==randomKey && testing_data_Sampled.size()<=test_data_Max)	//whenever there is a Random match, filter out this data
							{
							
								testing_data_Sampled.put(mapKey, (String) entry.getValue());//the random selection becomes "TEST" set
								itr.remove();//The remaining data, after removal becomes "TRAINING" set
								
								
								
							}
						
							
					}
					
					if(all_data.size()== training_data_Max){
						training_data_Sampled = all_data;
						generateNumBool = true;//stop the random number generation loop
					}
			}
			
			//extra part to write into file, to get train data
			FileWriter fWTrain=new FileWriter(WRITE_TRAIN_DATA_FILE_LOCATION);
			fWTrain.append(METADATA_CAR_DATA_SET);
			fWTrain.append(NEW_LINE_SEPARATOR);
			for(Map.Entry<Integer, String> kTr: training_data_Sampled.entrySet()){
				fWTrain.append(kTr.getValue());
				fWTrain.append(NEW_LINE_SEPARATOR);
			}
			fWTrain.close();
			
			//extra part to write into file, to get train data
			FileWriter fWTest=new FileWriter(WRITE_TEST_DATA_FILE_LOCATION);
			fWTest.append(METADATA_CAR_DATA_SET);
			fWTest.append(NEW_LINE_SEPARATOR);
			for(Map.Entry<Integer, String> kTst: testing_data_Sampled.entrySet()){
				fWTest.append(kTst.getValue());
				fWTest.append(NEW_LINE_SEPARATOR);
			}
			fWTest.close();
						
		}
		
		
		
	}

}
