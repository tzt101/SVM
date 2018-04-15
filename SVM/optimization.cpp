//
//  optimization.cpp
//  SVM
//
//  Created by tzt on 2018/4/12.
//  Copyright  2018 tzt. All rights reserved.
//

#include <iostream>
#include<fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iterator>
#include"classifier.h"
using namespace std;

int countLines(char *filename) {
	int n = 0;
	string tmp;
	ifstream name;
	name.open(filename);
	while (getline(name, tmp, '\n'))
	{
		n++;
	}
	name.close();
	return n;
};

template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
};
// define the data with x and t ...
struct data {
	vector<vector<float>> input;
	vector<float> label;
	vector<int> index;
	vector<float> loss;
	vector<float> weight;
	float accuracy;
};

struct data loadData(char *filename) {
	struct data dataset;
	int num;
	int size = 5; //the Dimensions+1 of x
	int each_label;
	num = countLines(filename);
	cout << "the number of train sample is: " << num << endl;
	ifstream name;
	name.open(filename);
	string middle;
	vector<float> m;
	for (int i = 0; i<num; i++) {
		m.clear();
		for (int j = 0; j<size; j++) {
			name >> middle;
			if (j == 0) {  //label
				each_label = stringToNum<float>(middle);
				if (each_label == 0) {
					dataset.label.push_back(-1);
				}
				else {
					dataset.label.push_back(each_label);
				}
			}
			else { //input
				m.push_back(stringToNum<float>(middle.substr(2, -1)));
			}
		}
		m.push_back(1);
		dataset.input.push_back(m);
	}
	name.close();
	return dataset;
};

void showSamples(struct data dataset, int n) {
	for (int i = 0; i<n; i++) {
		cout << "sample" << i << endl;
		cout << dataset.label.at(dataset.index.at(i)) << endl;
		for (int j = 0; j<5; j++) {
			cout << dataset.input[dataset.index.at(i)].at(j) << " ";
		}
		cout << endl;
	}
};

vector<int> shulleIndex(vector<int> index) {
	random_device rd;
	mt19937 g(rd());
	shuffle(index.begin(), index.end(), g);
	return index;
};

struct data next_batch(int i, int batch, int trainNum, struct data train_dataset) {
	struct data batch_train;
	int step = (i)*batch % trainNum;
	int begin;
	int end;
	if (step + batch - 1 < trainNum) {
		begin = step;
		end = step + batch;
		for (int i = begin; i<end; i++) {
			batch_train.label.push_back(train_dataset.label.at(train_dataset.index.at(i)));
			batch_train.input.push_back(train_dataset.input[train_dataset.index.at(i)]);
		}
	}
	else {
		//train_dataset.index = shulleIndex(train_dataset.index);
		begin = 0;
		end = batch;
		for (int i = begin; i<end; i++) {
			batch_train.label.push_back(train_dataset.label.at(train_dataset.index.at(i)));
			batch_train.input.push_back(train_dataset.input[train_dataset.index.at(i)]);
		}
	}
	return batch_train;
};

float meanLoss(vector<float> loss) {
	float m = 0;
	for (int i = 0; i<loss.size(); i++) {
		m += loss.at(i);
	}
	return m / loss.size();
};

void showWeight(vector<float> w) {
	cout << "the weight is: ";
	for (int i = 0; i<w.size(); i++) {
		cout << w.at(i) << " ";
	}
	cout << endl;
};


int main(int argc, const char * argv[]) {
	// insert code here...
	//SVM optimer;
	//read the train data
	char filename[] = "train.txt";
	char testfile[] = "test.txt";
	struct data train_dataset;
	struct data test_dataset;
	struct data batch_train;
	train_dataset = loadData(filename);
	test_dataset = loadData(testfile);
	int trainNum = countLines(filename);
	int batch_size = 5;
	int iter = 3089 / 5;
	float learning_rate = 0.005;
	for (int i = 0; i<trainNum; i++) {
		train_dataset.index.push_back(i);
	}
	//set optimizer
	SVM optimizer(learning_rate);
	optimizer.set_weight(1, 5);
	train_dataset.weight = optimizer.get_weight();
	showWeight(train_dataset.weight);
	//shuflle the idex of train sample
	train_dataset.index = shulleIndex(train_dataset.index);
	//showSamples(train_dataset, 3);
	for (int i = 0; i<iter; i++) {
		batch_train = next_batch(i, batch_size, trainNum, train_dataset);
		//cout << batch_train.input.size() << endl;
		batch_train.accuracy = optimizer.test(batch_train.input, batch_train.label);
		cout << "step" << i << "accuracy=" << batch_train.accuracy << endl;
		optimizer.forward(batch_train.input);
		optimizer.loss(batch_train.label);
		optimizer.gradient();
		optimizer.update();
		cout << "step " << i << ": loss= ";
		train_dataset.loss = optimizer.get_loss();
		train_dataset.weight = optimizer.get_weight();
		cout << meanLoss(train_dataset.loss) << endl;
	}
	train_dataset.weight = optimizer.get_weight();
	showWeight(train_dataset.weight);
	test_dataset.accuracy = optimizer.test(test_dataset.input, test_dataset.label);
	cout << test_dataset.accuracy << endl;


	return 0;
}
