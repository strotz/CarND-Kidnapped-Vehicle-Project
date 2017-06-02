/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <iomanip>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 100;

  particles.clear();
  weights.clear();

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  default_random_engine gen;
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // cout << "x=" << x << " ,y=" << y << endl;

  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    // cout << "particle " << i << " x=" << p.x << " ,y=" << p.y << endl;

    particles.push_back(p);
    weights.push_back(1.0);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;

  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  normal_distribution<double> dist_x(0, std_x);
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);

  for(int i = 0; i < particles.size(); ++i) { // TODO: use iterator

    Particle& p = particles[i];

    if (yaw_rate < 0.0001) {
      p.x += delta_t * velocity * cos(p.theta);
      p.y += delta_t * velocity * sin(p.theta);
      p.theta += yaw_rate * delta_t;

    } else {
      double theta_new = p.theta + delta_t * yaw_rate;
      double alfa =  velocity / yaw_rate;
      p.x += alfa * (sin(theta_new) - sin(p.theta));
      p.y += alfa * (cos(p.theta) - cos(theta_new));
      p.theta = theta_new;
    }

    // add noise
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }
}

inline double distance2(const LandmarkObs& one, const LandmarkObs& two) {
  double dx = one.x - two.x;
  double dy = one.y - two.y;
  return dx * dx + dy * dy;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  if (predicted.empty()) { // nothing to associate with
    return;
  }

  // for each measurement from sensors
  for(LandmarkObs& observation : observations) {

    int best_id = predicted[0].id;
    int best_distance = distance2(observation, predicted[0]);

    for(const LandmarkObs& prediction: predicted) {
      double d = distance2(prediction, observation);
      if (d < best_distance) {
        best_id = prediction.id;
        best_distance = d;
      }
    }
    observation.id = best_id; // associate id of nearest neighbor
  }
}

inline double distance2(const Particle& one, const Map::single_landmark_s& two) {
  double dx = one.x - two.x_f;
  double dy = one.y - two.y_f;
  return dx * dx + dy * dy;
}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  // useful pre-calculations
  double range_2 = sensor_range * sensor_range;
  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];
  double sigma_x_2 = 1 / (2 * sigma_x * sigma_x);
  double sigma_y_2 = 1 / (2 * sigma_y * sigma_y);
  double alfa = 1 / (2 * M_PI * sigma_x * sigma_y );

  int i = 0;
  for(Particle& particle: particles) {

    std::vector<LandmarkObs> predicted; // landmarks in map coordinates within sensor range to particle
    for(const Map::single_landmark_s& landmark: map_landmarks.landmark_list) {
      double d_2 = distance2(particle, landmark);
      if (d_2 < range_2) {
        LandmarkObs p;
        p.id = landmark.id_i;
        p.x = landmark.x_f;
        p.y = landmark.y_f;
        predicted.push_back(p);
      }
    }

    if (predicted.empty()) {
      // map does not have landmarks here, particle is useless
      particle.weight = 0.0;
      // cout << "nothing here" << endl;
    }
    else {

      std::vector<LandmarkObs> transformed_observations; // observations applied to particle and transferred to map coordinates
      for (const LandmarkObs &observation: observations) {
        double theta = particle.theta;

        LandmarkObs transferred;
        transferred.x = particle.x + observation.x * cos(theta) - observation.y * sin(theta);
        transferred.y = particle.y + observation.y * cos(theta) + observation.x * sin(theta);
        transformed_observations.push_back(transferred);
      }

      // TODO: it make sense to calculate Wi during data association, we know coordinates of landmark and observation
      dataAssociation(predicted, transformed_observations);

      // calculate weights
      particle.weight = 1.0;
      for (const LandmarkObs &observation: transformed_observations) {
        for (const Map::single_landmark_s &landmark: map_landmarks.landmark_list) {
          if (landmark.id_i == observation.id) {

            double dx = observation.x - landmark.x_f;
            double dy = observation.y - landmark.y_f;
            double prob = alfa * exp(-dx * dx * sigma_x_2 - dy * dy * sigma_y_2);
            particle.weight *= prob;
            break;
          }
        }
      }
    }

    weights[i++] = particle.weight;
  }
}

void ParticleFilter::resample() {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<int> d(weights.begin(), weights.end());

  // cout << endl;

  std::vector<Particle> result(num_particles);
  for(int i=0; i < num_particles; ++i) {
    int item = d(gen);
    // cout << "\r" << std::right << std::setw(7) << i << " item: " << std::right << std::setw(7) << item  << flush;
    result[i] = particles[item];
  }

  // cout << endl;

  std::copy(result.begin(), result.end(), particles.begin());
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
