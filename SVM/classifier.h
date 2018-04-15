//
//  classifier.h
//  SVM
//
//  Created by tzt on 2018/4/12.
//  Copyright  2018 tzt. All rights reserved.
//

#include<iostream>
#include<vector>
using namespace std;

class classifier {
public:
	vector<vector<float>> x;
	vector<float> y;
	vector<float> w;
	classifier();
	void set_x(vector<vector<float>> x);
	void set_y(vector<float> y);
	void set_weight(int set_model, int size);
	vector<float> get_weight();
	virtual void update() = 0;
};

class SVM :public classifier {
public:
	vector<float> l; //loss
	float s; //step
	vector<float> t; //label
	vector<vector<float>> g; //gradient
	SVM(float s);
	vector<float> get_loss();
	vector<float> get_forward();
	void forward(vector<vector<float>> x);
	void loss(vector<float> t);
	void gradient();
	void update();
	float test(vector<vector<float>> x, vector<float> t);

};
