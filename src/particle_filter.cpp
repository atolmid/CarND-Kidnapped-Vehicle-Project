#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

const int NUMBER_OF_PARTICLES = 100;
const double INITIAL_WEIGHT = 1.0;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
    //set number of particles that will be used
    num_particles = NUMBER_OF_PARTICLES;
    
    // Create normal distributions for x, y and theta.
    default_random_engine gen;    
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

    for(int i = 0; i<num_particles; i++){
    
    Particle p = {
    i,
    dist_x(gen),
    dist_y(gen),
    dist_theta(gen),
    INITIAL_WEIGHT
    };

    weights.push_back(1.0);
    particles.push_back(p);
  }

  is_initialized = true;
  return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	// Create normal distributions for noise x, y and theta.
	default_random_engine gen;

	normal_distribution<double> n_dist_x(0.0, std_pos[0]);
	normal_distribution<double> n_dist_y(0.0, std_pos[1]);
	normal_distribution<double> n_dist_theta(0.0, std_pos[2]);
		
  for(int i = 0; i<num_particles; i++){
    double x0 = particles[i].x;
    double y0 = particles[i].y;
    double theta0 = particles[i].theta;
    double theta1 = theta0 + yaw_rate * delta_t;
    
    if(fabs(yaw_rate) > 0.00001){
      particles[i].x = x0 + velocity/yaw_rate * (sin(theta1)-sin(theta0));
      particles[i].y = y0 + velocity/yaw_rate * (cos(theta0)-cos(theta1));
      particles[i].theta = theta1;
    }else{
      particles[i].x = x0 + velocity * sin(theta0) * delta_t;
      particles[i].y = y0 + velocity * cos(theta0) * delta_t;
      particles[i].theta = theta0;      
    }

    particles[i].x = particles[i].x + n_dist_x(gen);    
    particles[i].y = particles[i].y + n_dist_y(gen);    
    particles[i].theta = particles[i].theta + n_dist_theta(gen);    
  }	

  return;
  
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

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
	
	// Constants used for weight calculation
	const double sigmax = std_landmark[0]*std_landmark[0];
	const double sigmay = std_landmark[1]*std_landmark[1];
	const double d = 2 * M_PI * std_landmark[0] * std_landmark[1];
	
	//Clear the previous weights
	weights.clear();
	
  for(int i = 0; i<num_particles; i++){
  
    const double px = particles[i].x;
    const double py = particles[i].y;
    const double ptheta = particles[i].theta;
    
    //Clear the previous associations
    particles[i].associations.clear();
    particles[i].sense_x.clear();
    particles[i].sense_y.clear();
    
    double weight = INITIAL_WEIGHT;

    for(int j = 0; j<observations.size(); j++){
      // Transform observations to map coordinates
      double sense_x = px + observations[j].x * cos(ptheta) - observations[j].y * sin(ptheta);
      double sense_y = py + observations[j].x * sin(ptheta) + observations[j].y * cos(ptheta);
      // If outside the sensor's range, continue to the next one
      if(sqrt(pow(sense_x-px,2)+pow(sense_y-py,2)) > sensor_range) continue;
      
      particles[i].sense_x.push_back(sense_x);
      particles[i].sense_y.push_back(sense_y);
      // a big number for range 
      double min_range = 1000000000;
      // Index of landmark at minimum distance
      // Initialised at a negative number
      int min_id=-1;
      // Check map landmarks to find the one at the closest range
      for(int k = 0; k<map_landmarks.landmark_list.size(); k++){
        double l_x = map_landmarks.landmark_list[k].x_f;
        double l_y = map_landmarks.landmark_list[k].y_f;       
        double diff_x = l_x - sense_x;
        double diff_y = l_y - sense_y;
        double range = sqrt(pow(diff_x,2)+pow(diff_y,2));
        if(range < min_range){
          min_range = range;
          min_id = k;
        }
      }
      double l_x = map_landmarks.landmark_list[min_id].x_f;
      double l_y = map_landmarks.landmark_list[min_id].y_f;

      particles[i].associations.push_back(map_landmarks.landmark_list[min_id].id_i);
      
      // Compare observation by particle to observation by car, and update particle weight
      weight = weight * exp(-0.5 * ((l_x -sense_x)*(l_x - sense_x)/sigmax + (l_y - sense_y)*(l_y - sense_y)/sigmay)) / d;

    } 
    particles[i].weight=weight;
    weights.push_back(weight); 
  }		
  
  return;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
  vector<Particle> resampled_particles;
	  
  default_random_engine gen;
  discrete_distribution<int> index(weights.begin(), weights.end());

  for (int i = 0; i < NUMBER_OF_PARTICLES; i++) {

    int j = index(gen);
    resampled_particles.push_back(particles[j]);
    weights.push_back(particles[j].weight);
  }

  particles = resampled_particles;
  
  return;

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