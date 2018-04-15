//
//  classifier.cpp
//  SVM
//
//  Created by tzt on 2018/4/12.
//  Copyright  2018 tzt. All rights reserved.
//

#include <stdio.h>
#include "classifier.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <random>
using namespace std;

classifier::classifier() {
};

void classifier::set_x(vector<vector<float>> input) {
	x = input;
};

void classifier::set_y(vector<float> output) {
	y = output;
};

void classifier::set_weight(int model, int size) {
	// model should be set to 0:'zeros' or 1:'reandom'
	w.clear();
	if (model == 0) {
		for (int i = 0; i<size; i++) {
			w.push_back(0);
		}
	}
	else if (model == 1) {
		random_device rd;
		mt19937 g(rd());
		normal_distribution<double> n(0, 1);


		for (int i = 0; i<size; i++) {
			float a_w = n(g);
			w.push_back(a_w);
			//srand(time(0));
			//w.push_back(rand()%100/(double)101);
		}
	}
	//w.clear();
	//w.push_back(-1.89);w.push_back(-2.357653);w.push_back(3.9875);w.push_back(2.4566);w.push_back(1.98);
};

vector<float> classifier::get_weight() {
	return w;
};

void classifier::update() {
};

SVM::SVM(float lr) :classifier() {
	s = lr;
};

vector<float> SVM::get_forward() {
	return y;
};

vector<float> SVM::get_loss() {
	return l;
};

void SVM::forward(vector<vector<float>> input) {
	unsigned long int batch = input.size(); //batch_size
	unsigned long int len = w.size();       //the number of each sample
	y.clear();
	x.clear();
	x = input;
	for (int i = 0; i<batch; i++) {
		float m = 0;
		for (int j = 0; j<len; j++) {
			m += input[i].at(j) * w.at(j);
		}
		y.push_back(m);
	}
};

void SVM::loss(vector<float> label) {
	//t.clear();
	t = label;
	unsigned long int len = y.size();
	l.clear();
	for (int i = 0; i<len; i++) {
		float m = y.at(i) * label.at(i);
		if (m < 1) {
			l.push_back(1 - m);
		}
		else {
			l.push_back(0);
		}
	}
};

void SVM::gradient() {
	unsigned long int len = w.size();
	unsigned long int batch = x.size();
	g.clear();
	vector<float> m;
	for (int i = 0; i<batch; i++) {
		m.clear();
		for (int j = 0; j<len; j++) {
			if (l.at(i) > 0) {
				m.push_back(-t.at(i)*x[i].at(j));
			}
			else {
				m.push_back(0);
			}
		}
		g.push_back(m);
	}
};

void SVM::update() {
	unsigned long int len = w.size();
	unsigned long int batch = x.size();
	float m;
	for (int i = 0; i<len; i++) {
		m = 0;
		for (int j = 0; j<batch; j++) {
			m += g[j].at(i);
		}
		//cout << s << endl;
		//cout << s*m/batch << endl;
		w.at(i) -= s*m / batch;
	}
};

float SVM::test(vector<vector<float>> input, vector<float> label) {
	x.clear();
	t.clear();
	//y.clear();
	x = input;
	t = label;
	unsigned long int len = w.size();
	unsigned long int batch = x.size();
	float correct = 0;
	for (int i = 0; i<batch; i++) {
		float m = 0;
		for (int j = 0; j<len; j++) {
			m += x[i].at(j) * w.at(j);
		}

		if (m*t.at(i) >= 0) {
			//cout << i << endl;
			correct += 1;
		}
	}
	correct = correct / batch;
	return correct;
};


