
#include <iostream>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

// Define the Ising class
class Ising {

public:
    // Constructor
    Ising(int n_, double temp_);

    // Methods
    void initialize();
    void flip_spin(int i, int j);
    double get_spin(int i, int j);
    double calc_energy();
    double calc_magnetization();
    void print_lattice();

private:
    // Attributes
    int n;
    double temp;
    vector< vector<double> > lattice;
};

// Constructor for the Ising class
Ising::Ising(int n_, double temp_) {
    n = n_;
    temp = temp_;
}

// Initialize the lattice with random spins
void Ising::initialize() {

    // Create a uniform random number generator
    random_device rd;  //Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    uniform_real_distribution<> dis(0, 1);

    // Loop over all spins and set them randomly
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (dis(gen) < 0.5) {
                lattice[i][j] = 1;
            }
            else {
                lattice[i][j] = -1;
            }
        }
    }
}

// Flip the spin at index (i,j)
void Ising::flip_spin(int i, int j) {
    lattice[i][j] *= -1;
}

// Get the spin at index (i,j)
double Ising::get_spin(int i, int j) {
    return lattice[i][j];
}

// Calculate the energy of the lattice
double Ising::calc_energy() {

    double energy = 0;

    // Loop over all nearest neighbor pairs and calculate the energy
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            energy += -1 * get_spin(i,j) * (get_spin(i+1,j) + get_spin(i,j+1));
        }
    }

    return energy;
}

// Calculate the magnetization of the lattice
double Ising::calc_magnetization() {

    double magnetization = 0;

    // Loop over all spins and calculate the magnetization
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            magnetization += get_spin(i,j);
        }
    }

    return magnetization;
}

// Print the lattice to the screen
void Ising::print_lattice() {

    // Loop over all spins and print them to the screen
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cout << get_spin(i,j) << " ";
        }
        cout << endl;
    }
}

int main() {

    // Set the size of the lattice
    int n = 20;

    // Set the temperature
    double temp = 1.0;

    // Create the Ising model
    Ising model(n, temp);

    // Initialize the lattice
    model.initialize();

    // Print the initial lattice
    cout << "Initial lattice:" << endl;
    model.print_lattice();

    // Calculate the initial energy and magnetization
    double energy = model.calc_energy();
    double magnetization = model.calc_magnetization();

    // Print the initial energy and magnetization
    cout << "Initial energy: " << energy << endl;
    cout << "Initial magnetization: " << magnetization << endl;

    return 0;
}