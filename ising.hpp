#ifndef ISING_HPP
#define ISING_HPP

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>
#include <numeric>

// Negative number proof modulo
template <typename T> T mod(T a, T b) {
  T ret = a % b;
  while (ret < 0) {
    ret += b;
  }
  return ret;
}

class Ising {
public:
  // Constructor
  Ising(double beta, int length) {
    // Seed the random number generator
    m_rng.seed(42);

    // Store parameters
    m_beta = beta;
    m_length = length;

    // Setup data field to a random state
    m_data.resize(length * length);

    for (auto &spin : m_data) {
      // first randomly choose 0 or 1
      spin = random_int(2) * 2;
      // turn the 0 or 2 into -1 or 1
      spin -= 1; // turn 0 into -1 and 2 into 1
      assert((spin == 1) || (spin == -1));
    };

    // Initialize running energy and magnetization
    update();
  };

  // Recalculate the magnetization from the state of the Ising model
  void recalculate_magnetization() {
    // the magnetization is just the sum of all spins
    // it is normalized later in get_magnetization
    m_Magnetization = std::accumulate(
      m_data.begin(),
      m_data.end(),
      0
    );
    assert((m_Magnetization >= -m_length * m_length) && (m_Magnetization <= m_length * m_length));
  };

  // Recalculate the energy from the state of the Ising model
  void recalculate_energy() {
    // set the energy to 0
    m_Energy = 0;
    // loop over all lattice sites
    for (int row = 0; row < m_length; row++) {
      for (int column = 0; column < m_length; column++) {
        // add the respective energy to m_Energy
        m_Energy += lattice_energy(row, column);
      }
    };
    // divide the result by 2
    m_Energy /= 2;
  };

  // calculate the energy associated with a single lattice side
  double lattice_energy(int row, int column) {
    // create a vector with all the neighboring spins
    const std::vector<int> neighboring_spins = {
      get_spin(mod(row-1, m_length), column),
      get_spin(mod(row+1, m_length), column),
      get_spin(row, mod(column-1, m_length)),
      get_spin(row, mod(column+1, m_length))
    };
    // calculate the respective energy
    const double energy = - get_spin(row, column) * std::accumulate(
      neighboring_spins.begin(),
      neighboring_spins.end(),
      0
    );
    return energy;
  };

  // Update running values for energy and magnetization
  void update() {
    recalculate_energy();
    recalculate_magnetization();
    m_invalid = false;
  };

  // Get spin at row,colhmn
  int get_spin(int row, int column) {
    const int _row = mod(row, m_length);
    const int _column = mod(column, m_length);
    assert(_row >= 0 && _row < m_length);
    assert(_column >= 0 && _column < m_length);
    return m_data[get_linear_index(_row, _column)];
  };

  // Set spin at row,column
  void set_spin(int row, int column, int new_spin) {
    assert((new_spin == 1) || (new_spin == -1));
    const int _row = mod(row, m_length);
    const int _column = mod(column, m_length);
    assert(_row >= 0 || _row < m_length);
    assert(_column >= 0 && _column < m_length);
    m_data[get_linear_index(_row, _column)] = new_spin;
    m_invalid = true;
  };

  // Try to flip a spin and return true if move accepted,
  // update magnetization/energy if move was accepted
  bool try_flip(int row, int column) {
    // get the old spin
    const int _row = mod(row, m_length);
    const int _column = mod(column, m_length);
    assert((_row >= 0) && (_row < m_length));
    assert((_column >= 0) && (_column < m_length));
    const int old_spin = get_spin(_row, _column);
    assert((old_spin == 1) || (old_spin == -1));
    // get the new spin and the energy difference
    const int new_spin = -1 * old_spin;
    const double energy_difference = -2 * lattice_energy(_row, _column);
    // decide whether to accept the flip
    const bool accept = (
      random_double() < std::min(
        static_cast<double>(1),
        std::exp(-m_beta * energy_difference)
      )
    );
    // update m_data and m_Energy
    if (accept) {
      set_spin(_row, _column, new_spin);
      // update();
      m_Energy += energy_difference;
      m_invalid = true;
    }
    return accept;
  };

  // Try flipping a random spin. Return true if move accepted
  // If move was accepted, also update energy and magnetization
  bool try_random_flip() {
    // get a random row and column
    const int row = random_int(m_length);
    const int column = random_int(m_length);
    assert((row >= 0) || (row < m_length));
    assert((column >= 0) || (column < m_length));
    // try the flip
    const bool accept = try_flip(row, column);
    return accept;
  };

  void try_many_random_flips(int number_of_flips) {
    for (int flip = 0; flip < number_of_flips; flip++) {
      try_random_flip();
    }
  };

  // Get the current energy
  double get_energy() {
    if (m_invalid) {
      update();
    }
    return m_Energy;
  };

  // Get the current magnetization
  double get_magnetization() {
    if (m_invalid) {
      update();
    }
    return m_Magnetization / (m_length * m_length);
  };

  // Get the raw data
  std::vector<int> get_data() { return m_data; };

private:
  // Parameters
  double m_beta;
  int m_length;

  // Running magnetization, energy
  double m_Magnetization;
  double m_Energy;
  // Are the running values invalid
  bool m_invalid;

  // Ising model
  std::vector<int> m_data;

  // Random number generator
  std::mt19937 m_rng;
  std::uniform_real_distribution<double> m_uniform_dist;
  double random_double() { return m_uniform_dist(m_rng); }

  int random_int(int below) {
    auto x = random_double();
    auto v = static_cast<int>(x * below);
    assert(v >= 0 && v < below);
    return v;
  };

  // Convert 2d indices from 0<=i,j<l to linear index 0<ind<l*l
  int get_linear_index(int row, int column) {
    // the linear index rounts each row from the left to the right
    // starting with the first row
    int index = m_length * row + column;
    assert((index < m_length * m_length) && (index >= 0));
    return index;
  };
};

#endif
