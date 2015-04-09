// This file provides fast analysis functions written in c++. This file is needed to circumvent some python limitations where
// no sufficient pythonic solution is available.
#pragma once

#include <iostream>
#include <string>
#include <ctime>
#include <cmath>
#include <exception>
#include <algorithm>
#include <sstream>

#include "Basis.h"
#include "defines.h"

// counts from the event number column of the cluster table how often a cluster occurs in every event
unsigned int getNclusterInEvents(int64_t*& rEventNumber, const unsigned int& rSize, int64_t*& rResultEventNumber, unsigned int*& rResultCount)
{
	unsigned int tResultIndex = 0;
	unsigned int tLastIndex = 0;
	int64_t tLastValue = 0;
	for (unsigned int i = 0; i < rSize; ++i) {  // loop over all events can count the consecutive equal event numbers
		if (i == 0)
			tLastValue = rEventNumber[i];
		else if (tLastValue != rEventNumber[i]) {
			rResultCount[tResultIndex] = i - tLastIndex;
			rResultEventNumber[tResultIndex] = tLastValue;
			tLastValue = rEventNumber[i];
			tLastIndex = i;
			tResultIndex++;
		}
	}
	// add last event
	rResultCount[tResultIndex] = rSize - tLastIndex;
	rResultEventNumber[tResultIndex] = tLastValue;
	return tResultIndex + 1;
}

//takes two event arrays and calculates an intersection array of event numbers occurring in both arrays
unsigned int getEventsInBothArrays(int64_t*& rEventArrayOne, const unsigned int& rSizeArrayOne, int64_t*& rEventArrayTwo, const unsigned int& rSizeArrayTwo, int64_t*& rEventArrayIntersection)
{
	int64_t tActualEventNumber = -1;
	unsigned int tActualIndex = 0;
	unsigned int tActualResultIndex = 0;
	for (unsigned int i = 0; i < rSizeArrayOne; ++i) {  // loop over all event numbers in first array
		if (rEventArrayOne[i] == tActualEventNumber)  // omit the same event number occuring again
			continue;
		tActualEventNumber = rEventArrayOne[i];
		for (unsigned int j = tActualIndex; j < rSizeArrayTwo; ++j) {
			if (rEventArrayTwo[j] >= tActualEventNumber) {
				tActualIndex = j;
				break;
			}
		}
		if (rEventArrayTwo[tActualIndex] == tActualEventNumber) {
			rEventArrayIntersection[tActualResultIndex] = tActualEventNumber;
			tActualResultIndex++;
		}
	}
	return tActualResultIndex++;
}

//takes two event number arrays and returns a event number array with the maximum occurrence of each event number in array one and two
unsigned int getMaxEventsInBothArrays(int64_t*& rEventArrayOne, const unsigned int& rSizeArrayOne, int64_t*& rEventArrayTwo, const unsigned int& rSizeArrayTwo, int64_t*& result, const unsigned int& rSizeArrayResult)
{
	int64_t tFirstActualEventNumber = rEventArrayOne[0];
	int64_t tSecondActualEventNumber = rEventArrayTwo[0];
	int64_t tFirstLastEventNumber = rEventArrayOne[rSizeArrayOne - 1];
	int64_t tSecondLastEventNumber = rEventArrayTwo[rSizeArrayTwo - 1];
	unsigned int i = 0;
	unsigned int j = 0;
	unsigned int tActualResultIndex = 0;
	unsigned int tFirstActualOccurrence = 0;
	unsigned int tSecondActualOccurrence = 0;

	bool first_finished = false;
	bool second_finished = false;

//	std::cout<<"tFirstActualEventNumber "<<tFirstActualEventNumber<<std::endl;
//	std::cout<<"tSecondActualEventNumber "<<tSecondActualEventNumber<<std::endl;
//	std::cout<<"tFirstLastEventNumber "<<tFirstLastEventNumber<<std::endl;
//	std::cout<<"tSecondLastEventNumber "<<tSecondLastEventNumber<<std::endl;
//	std::cout<<"rSizeArrayOne "<<rSizeArrayOne<<std::endl;
//	std::cout<<"rSizeArrayTwo "<<rSizeArrayTwo<<std::endl;
//	std::cout<<"rSizeArrayResult "<<rSizeArrayResult<<std::endl;

	while (!(first_finished && second_finished)) {
		if ((tFirstActualEventNumber <= tSecondActualEventNumber) || second_finished) {
			unsigned int ii;
			for (ii = i; ii < rSizeArrayOne; ++ii) {
				if (rEventArrayOne[ii] == tFirstActualEventNumber)
					tFirstActualOccurrence++;
				else
					break;
			}
			i = ii;
		}

		if ((tSecondActualEventNumber <= tFirstActualEventNumber) || first_finished) {
			unsigned int jj;
			for (jj = j; jj < rSizeArrayTwo; ++jj) {
				if (rEventArrayTwo[jj] == tSecondActualEventNumber)
					tSecondActualOccurrence++;
				else
					break;
			}
			j = jj;
		}

//		std::cout<<"tFirstActualEventNumber "<<tFirstActualEventNumber<<" "<<tFirstActualOccurrence<<" "<<first_finished<<std::endl;
//		std::cout<<"tSecondActualEventNumber "<<tSecondActualEventNumber<<" "<<tSecondActualOccurrence<<" "<<second_finished<<std::endl;

		if (tFirstActualEventNumber == tSecondActualEventNumber) {
//			std::cout<<"==, add "<<std::max(tFirstActualOccurrence, tSecondActualOccurrence)<<" x "<<tFirstActualEventNumber<<std::endl;
			if (tFirstActualEventNumber == tFirstLastEventNumber)
				first_finished = true;
			if (tSecondActualEventNumber == tSecondLastEventNumber)
				second_finished = true;
			for (unsigned int k = 0; k < std::max(tFirstActualOccurrence, tSecondActualOccurrence); ++k) {
				if (tActualResultIndex < rSizeArrayResult)
					result[tActualResultIndex++] = tFirstActualEventNumber;
				else
					throw std::out_of_range("The result histogram is too small. Increase size.");
			}
		}
		else if ((!first_finished && tFirstActualEventNumber < tSecondActualEventNumber) || second_finished) {
//			std::cout<<"==, add "<<tFirstActualOccurrence<<" x "<<tFirstActualEventNumber<<std::endl;
			if (tFirstActualEventNumber == tFirstLastEventNumber)
				first_finished = true;
			for (unsigned int k = 0; k < tFirstActualOccurrence; ++k) {
				if (tActualResultIndex < rSizeArrayResult)
					result[tActualResultIndex++] = tFirstActualEventNumber;
				else
					throw std::out_of_range("The result histogram is too small. Increase size.");
			}
		}
		else if ((!second_finished && tSecondActualEventNumber < tFirstActualEventNumber) || first_finished) {
//			std::cout<<"==, add "<<tSecondActualOccurrence<<" x "<<tSecondActualEventNumber<<std::endl;
			if (tSecondActualEventNumber == tSecondLastEventNumber)
				second_finished = true;
			for (unsigned int k = 0; k < tSecondActualOccurrence; ++k) {
				if (tActualResultIndex < rSizeArrayResult)
					result[tActualResultIndex++] = tSecondActualEventNumber;
				else
					throw std::out_of_range("The result histogram is too small. Increase size.");
			}
		}

		if (i < rSizeArrayOne)
			tFirstActualEventNumber = rEventArrayOne[i];
		if (j < rSizeArrayTwo)
			tSecondActualEventNumber = rEventArrayTwo[j];
		tFirstActualOccurrence = 0;
		tSecondActualOccurrence = 0;
	}

	return tActualResultIndex;
}

//does the same as np.in1d but uses the fact that the arrays are sorted
void in1d_sorted(int64_t*& rEventArrayOne, const unsigned int& rSizeArrayOne, int64_t*& rEventArrayTwo, const unsigned int& rSizeArrayTwo, uint8_t*& rSelection)
{
	rSelection[0] = true;
	int64_t tActualEventNumber = -1;
	unsigned int tActualIndex = 0;
	for (unsigned int i = 0; i < rSizeArrayOne; ++i) {  // loop over all event numbers in first array
		tActualEventNumber = rEventArrayOne[i];
		for (unsigned int j = tActualIndex; j < rSizeArrayTwo; ++j) {
			if (rEventArrayTwo[j] >= tActualEventNumber) {
				tActualIndex = j;
				break;
			}
		}
		if (rEventArrayTwo[tActualIndex] == tActualEventNumber)
			rSelection[i] = 1;
		else
			rSelection[i] = 0;
	}
}

// fast 1d index histograming (bin size = 1, values starting from 0)
void histogram_1d(int*& x, const unsigned int& rSize, const unsigned int& rNbinsX, uint32_t*& rResult)
{
	for (unsigned int i = 0; i < rSize; ++i) {
		if (x[i] >= rNbinsX)
			throw std::out_of_range("The histogram indices are out of range");
		if (rResult[x[i]] < 4294967295)
			++rResult[x[i]];
		else
			throw std::out_of_range("The histogram has more than 4294967295 entries per bin. This is not supported.");
	}
}

// fast 2d index histograming (bin size = 1, values starting from 0)
void histogram_2d(int*& x, int*& y, const unsigned int& rSize, const unsigned int& rNbinsX, const unsigned int& rNbinsY, uint32_t*& rResult)
{
	for (unsigned int i = 0; i < rSize; ++i) {
		if (x[i] >= rNbinsX || y[i] >= rNbinsY)
			throw std::out_of_range("The histogram indices are out of range");
		if (rResult[x[i] * rNbinsY + y[i]] < 4294967295)
			++rResult[x[i] * rNbinsY + y[i]];
		else
			throw std::out_of_range("The histogram has more than 4294967295 entries per bin. This is not supported.");
	}
}

// fast 3d index histograming (bin size = 1, values starting from 0)
void histogram_3d(int*& x, int*& y, int*& z, const unsigned int& rSize, const unsigned int& rNbinsX, const unsigned int& rNbinsY, const unsigned int& rNbinsZ, uint16_t*& rResult)
{
	for (unsigned int i = 0; i < rSize; ++i) {
		if (x[i] >= rNbinsX || y[i] >= rNbinsY || z[i] >= rNbinsZ) {
			std::stringstream errorString;
			errorString << "The histogram indices (x/y/z)=(" << x[i] << "/" << y[i] << "/" << z[i] << ") are out of range.";
			throw std::out_of_range(errorString.str());
		}
		if (rResult[x[i] * rNbinsY * rNbinsZ + y[i] * rNbinsZ + z[i]] < 65535)
			++rResult[x[i] * rNbinsY * rNbinsZ + y[i] * rNbinsZ + z[i]];
		else
			throw std::out_of_range("The histogram has more than 65535 entries per bin. This is not supported.");
	}
}

// fast mapping of cluster hits to event numbers
void mapCluster(int64_t*& rEventArray, const unsigned int& rEventArraySize, ClusterInfo*& rClusterInfo, const unsigned int& rClusterInfoSize, ClusterInfo*& rMappedClusterInfo, const unsigned int& rMappedClusterInfoSize)
{
	unsigned int j = 0;
	for (unsigned int i = 0; i < rEventArraySize; ++i) {
		for (j; j < rClusterInfoSize; ++j) {
			if (rClusterInfo[j].eventNumber == rEventArray[i]) {
				if (i < rEventArraySize) {
					rMappedClusterInfo[i] = rClusterInfo[j];
					++i;
				}
				else
					return;
			}
			else
				break;
		}
	}
}

// loop over the refHit, Hit arrays and compare the hits of same event number. If they are similar (within an error) correlation is assumed. If more than events are not correlated, broken correlation is assumed.
bool _checkForCorrelation(unsigned int& iRefHit, unsigned int& iHit, const int64_t*& rEventArray, const double*& rRefCol, double*& rCol, const double*& rRefRow, double*& rRow, const unsigned int& nHits, const double& rError, const unsigned int& nBadEvents)
{
	int64_t tEventNumber = 0;  // last read event number
	unsigned int tBadEvents = 0;  // consecutive not correlated events
	unsigned int tHitIndex = 0;  // actual first not correlated hit index
	bool tIsCorrelated = false;  // flag for the actual event

	for (iRefHit; iRefHit < nHits; ++iRefHit) {
		if (tEventNumber != rEventArray[iRefHit]) {
			if (!tIsCorrelated) {
				if (tBadEvents == 0){
					tHitIndex = iRefHit;
					for(tHitIndex; tHitIndex >= 0; --tHitIndex){  // the actual first not correlated hit is the first hit from the event before
						if (rEventArray[tHitIndex] == tEventNumber - 1){
							tHitIndex++;
							break;
						}
					}
				}
				tBadEvents++;
			}
			else
				tBadEvents = 0;
			if (tBadEvents >= nBadEvents) {  // a correlation is defined as broken if more than nBadEvents consecutive not correlated events exist
				iRefHit = tHitIndex;  // set reference hit to first not correlated hit
				return false;
			}
			tEventNumber = rEventArray[iRefHit];
			tIsCorrelated = false;
		}
//		std::cout << iRefHit << "\t" << rEventArray[iRefHit] << "\t" << rRefCol[iRefHit] << " / " << rCol[iRefHit] << "\t" << rRefRow[iRefHit] << " / " << rRow[iRefHit] << "\t" << tBadEvents << "\n";
		if (rRefCol[iRefHit] == 0 || rCol[iRefHit] == 0 || rRefRow[iRefHit] == 0 || rRow[iRefHit] == 0)  // no hit (col = row = 0) means no correlation
			continue;
		if (std::fabs(rRefCol[iRefHit] - rCol[iRefHit]) < rError && std::fabs(rRefRow[iRefHit] - rRow[iRefHit]) < rError)
			tIsCorrelated = true;

	}

	return true;
}

void _correctAlignment(unsigned int i, const int64_t*& rEventArray, const double*& rRefCol, double*& rCol, const double*& rRefRow, double*& rRow, const unsigned int& nHits, const double& rError)
{
	unsigned int consHitSearchDistance = 5;  // the search distance for subsequent correlated hits

	// find first not correlated reference hit
	double tActualColumn = rRefCol[i];
	double tActualRow = rRefRow[i];
	for (unsigned int j = 0; j < nHits; ++j) {  // hit has to be non virtual (column/row != 0)
		if (tActualColumn == 0 && tActualRow == 0) {
			tActualColumn = rRefCol[i + j];
			tActualRow = rRefRow[i + j];
		}
		else
			break;
	}
	std::cout << "Try to find hit for " << i << " " << rEventArray[i] << " " << tActualColumn << " " << tActualRow << "\n";

	// Status variables
	bool tFoundFittingHit = false;  // flag that a correlated hit candidate is found
	unsigned int tFittingHitIndex = 0;  // the index of the correlated hit candidate
	unsigned int tOtherRefHitIndex = 0; // index of the following reference hits that are checked for correlation
	unsigned int tSearchDistance = 10000; // search range (index) after not correlated hit
	unsigned int tConsecutiveCorrHits = 0;  // if correlated hit is found check subsequent hits if they are still correlated, to suppress correlation by chance

	// Determine the search distance for the correlated hit
	unsigned int tStartSearchIndex = i;
	unsigned int tStopSearchIndex = nHits;
	if (i + tSearchDistance < nHits)
		tStopSearchIndex = i + tSearchDistance;
	std::cout << "Search between " << tStartSearchIndex << " and " << tStopSearchIndex << "\n";

	// Loop over the hits within the search distance and try to find a fitting hit. All fitting hits are checked to have subsequent correlated hits. Otherwise it is only correlation by chance.
	for (unsigned int j = tStartSearchIndex; j < tStopSearchIndex; ++j) {
		// Check if subsequent hits of hit candidate are also correlated
		tOtherRefHitIndex = i + j - tFittingHitIndex;
//		std::cout<<"i "<<i<<" j "<<j<<" tOtherRefHitIndex "<<tOtherRefHitIndex<<"\n";
		if (tFoundFittingHit && rCol[j] != 0 && std::fabs(rRefCol[tOtherRefHitIndex] - rCol[j]) < rError && std::fabs(rRefRow[tOtherRefHitIndex] - rRow[j]) < rError) {  // check if following hits are still correlated
			tConsecutiveCorrHits++;
			std::cout << "Found " << tConsecutiveCorrHits << ". subsequent correlated hit " << rEventArray[j] << " " << rCol[j] << " " << rRow[j] << " for " << rEventArray[tOtherRefHitIndex] << " " << rRefCol[tOtherRefHitIndex] << " " << rRefRow[tOtherRefHitIndex] << "\n";
		}
		if (tFoundFittingHit && j - tFittingHitIndex > consHitSearchDistance) {
			if (tConsecutiveCorrHits > 2) {  // the fitting hit was successfully found if 2 / consHitSearchDistance hits are correlated
				std::cout << "Found jump with index offset = " << tFittingHitIndex - i << " at reference hit index " << i << "\n";
				break;
			}
			else {
				std::cout << "Hit candidate failed\n";
				tFoundFittingHit = false;
			}
		}

//		std::cout<<"rCol[j] "<<rCol[j]<<" rRow[j] "<<rRow[j]<<"\n";
		// Search for correlated hit candidate
		if (std::fabs(tActualColumn - rCol[j]) < rError && std::fabs(tActualRow - rRow[j]) < rError) {  // check for correlation
			std::cout << "Found correlated hit canditate " << j << " " << rEventArray[j] << " " << rCol[j] << " " << rRow[j] << "\n";
			tFoundFittingHit = true;
			tFittingHitIndex = j;
			for (unsigned int k = 0; k < 20; ++k) {
				std::cout << "Next " << k << " " << rEventArray[j + k] << " " << rCol[j + k] << "/" << rRow[j + k] << " = " << rEventArray[i + k] << " " << rRefCol[i + k] << "/" << rRefRow[i + k] << "\n";
			}
		}
	}
}

// Fix the event alignment with hit position information, crazy...
void fixEventAlignment(const int64_t*& rEventArray, const double*& rRefCol, double*& rCol, const double*& rRefRow, double*& rRow, const unsigned int& nHits, const double& rError, const unsigned int& nBadEvents)
{
	unsigned int iRefHit = 0;
	unsigned int iHit = 0;
	if (!_checkForCorrelation(iRefHit, iHit,rEventArray, rRefCol, rCol, rRefRow, rRow, nHits, rError, nBadEvents))
		std::cout<<"No correlation starting at index/event: "<<iRefHit<<"/"<<rEventArray[iRefHit]<<"\n";
	else
		std::cout<<"Everything is correlated!\n";
//	int64_t tEventNumber = 0;  // last read event number
//	unsigned int tBadEvents = 0;  // consecutive not correlated events
//	unsigned int tHitIndex = 0;  // actual first not correlated hit index
//	bool tIsCorrelated = false;  // flag for the actual event
//	for (unsigned int i = 0; i < nHits; ++i) {
//		if (tEventNumber != rEventArray[i]) {
////			if (rEventArray[i] < 2500)
////				std::cout<<rEventArray[i]<<"\t"<<isCorrelated<<"\t"<<tBadEvents<<"\n";
////			if (!tIsCorrelated)
////				std::cout<<rEventArray[i]<<" is not correlated \n";
//			tEventNumber = rEventArray[i];
//			if (!tIsCorrelated) {
//				if (tBadEvents == 0)
//					tHitIndex = i - 1;
//				tBadEvents++;
//			}
//			else
//				tBadEvents = 0;
//			if (tBadEvents >= nBadEvents) {  // a correlation is defined as broken if more than nBadEvents consecutive not correlated events exist
//				_correctAlignment(tHitIndex, rEventArray, rRefCol, rCol, rRefRow, rRow, nHits, rError);
////				std::cout<<rEventArray[i]<<"\t"<<tIsCorrelated<<"\t"<<tBadEvents<<"\n";
//				break;
//			}
//			tIsCorrelated = false;
//		}
//		if (rEventArray[i] > 7140 && rEventArray[i] < 7180)
//			std::cout << i << "\t" << rEventArray[i] << "\t" << rRefCol[i] << " / " << rCol[i] << "\t" << rRefRow[i] << " / " << rRow[i] << "\t" << tBadEvents << "\n";
//		if (rRefCol[i] == 0 || rCol[i] == 0 || rRefRow[i] == 0 || rRow[i] == 0)  // no hit (col = row = 0) means no correlation
//			continue;
//		if (std::fabs(rRefCol[i] - rCol[i]) < rError && std::fabs(rRefRow[i] - rRow[i]) < rError)
//			tIsCorrelated = true;
//
//	}
}

