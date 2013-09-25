// ------------------------------------------------------------------------------------------------
// C++ Implementation of Crank-Nicolson finite difference method for American/European option pricing.
// Requires Boost 1.5.4
// Author: Ethan Li
// Based off implementation by Dr. Fab at Volopta.com
// ------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <ctime>

namespace ublas = boost::numeric::ublas;

using namespace System;
using namespace std;

enum PutCall{
	Put,
	Call
};

enum OptionType{
	European,					//European
	American					//American
};

//Option features

const double T		  	= 1;	// Maturity
const double Spot	  	= 100;	// Spot price
const double K		  	= 100;  // Strike Price
const double v		  	= 0.2;	// Volatility
const double r		  	= 0.06;	// Risk-free rate
const double q		  	= 0.03; // Dividend yield;
const PutCall putCall 	= Put;	
const OptionType oType	= American;

//PDE features
const int N			= 500;					// Number of time steps
const int M			= 500;					// Number of stock price steps
const double dt			= T/N;					// Time increment
const double mu			= r - q - (v*v) / 2;	// Drift for stock process
const double dx			= v * sqrt(3 * dt);		// Increment for stock price

//Probabilities
const double pu = -0.25 * dt * ((v * v) / (dx * dx) + mu / dx);      //Up probability
const double pm = 1.0 + dt * (v * v) / 2.0 / (dx * dx) + r*dt / 2.0;      //Middle probability
const double pd = -0.25 * dt*((v * v) / (dx * dx) - mu / dx);      //Down probability



int main()
{
	using namespace ublas;
	clock_t begin = clock();

	//Initialize stock price and option values.
	ublas::vector<double> S(2 * M + 1, 0);
	matrix<double> V(2 * M + 1, N + 1, 0);

	// Indices for stock price step
	ublas::vector<double> J(2 * M + 1);
	for (int i = M; i >= -M; --i)
		J(M - i) = i;

	// Stock price at maturity
	for (unsigned i = 0; i < S.size(); ++i)
		S(i) = Spot * exp(J(i) * dx);

	// Option price at maturity
	for (unsigned i = 0; i < V.size1(); i++)
		V(i, V.size2()-1) = ((putCall == Put) ? max(K - S(i), 0.0) : max(S(i) - K, 0.0)); //populate last column of V
	

	//Initialize required matrices
	matrix<double> pmp(2 * M + 1, N + 1, 0);
	matrix<double> pp(2 * M + 1, N + 1, 0);
	matrix<double> C(2 * M + 1, N + 1, 0);

	
	//Create the P' matrix
	for (unsigned j = N; j > 0; --j)
	{
		pmp(2 * M-1, j) = pd + pm;
		for (unsigned i = 2 * M - 2; i > 0; --i)
			pmp(i, j) = pm - pd / (pmp(i + 1, j)) * pu;
	}


	//Upper and lower bounds for the American put
	double lambda_L = (putCall == Put) ? max(K - S(2 * M), 0.0) : 0.0;
	double lambda_U = (putCall == Call) ? max(S(0) - K, 0.0) : 0.0;

	// Work backwards and obtain matrix of values V
	for (unsigned j = N; j > 0; --j)
	{
		pp(2 * M - 1, j) = -pu * V(2 * M - 2, j) - (pm - 2)*V(2 * M - 1, j) - pd*V(2 * M, j) + pd*lambda_L;

		for (unsigned i = 2 * M - 2; i > 0; --i)
		{
			pp(i,j) = -pu * V(i - 1, j) - (pm - 2)*V(i, j) - pd*V(i + 1, j) - pd / pmp(i + 1, j) * pp(i + 1, j);
		}

		for (unsigned i = 0; i < 2 * M + 1; ++i)
		{
			if (i == 0)
				C(i, j-1) = (pp(i + 1, j) + pmp(i + 1, j) * lambda_U) / (pmp(i + 1, j) + pu);
			else if (i < 2 * M)
				C(i, j - 1) = (pp(i, j) - pu*C(i - 1, j - 1)) / pmp(i, j);
			else
				C(i, j - 1) = C(i - 1, j - 1) - lambda_L;

			if (oType == American)
				V(i, j - 1) = max((putCall == Put) ? K - S(i) : S(i) - K, C(i, j - 1));
			else
				V(i, j - 1) = C(i, j - 1);
		}


	}
	
	clock_t end = clock();

	double secs = double(end - begin) / CLOCKS_PER_SEC;

	cout << "Option price: "<< V(M,0) << endl;
	cout << "Elapsed seconds: " << secs << endl;


	return 0;
}
