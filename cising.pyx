import numpy as np
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared
from libcpp cimport bool

# declare the interface to the C code
cdef extern from "ising.hpp":
    cppclass Ising:
        Ising(double beta, int length)
        int get_spin(int row, int column)
        void set_spin(int row, int column, int new_spin)
        double get_energy()
        double get_magnetization()
        vector[int] get_data()
        bool try_random_flip()
        void try_many_random_flips(int number_of_flips)


cdef class IsingModel:
  cdef shared_ptr[Ising] c_ising
  cdef int length

  def __init__(self, double beta, int length):
    self.length = length
    self.c_ising = make_shared[Ising](beta, length)

  def as_numpy(self):
    return np.array(self.c_ising.get().get_data()).reshape((self.length, self.length))

  def get_spin(self, row, column):
    return self.c_ising.get().get_spin(row, column)

  def set_spin(self, row, column, new_spin):
    self.c_ising.get().set_spin(row, column, new_spin)

  def energy(self):
    return self.c_ising.get().get_energy()

  def magnetization(self):
    return self.c_ising.get().get_magnetization()

  def try_random_flip(self):
    self.c_ising.get().try_random_flip()

  def try_many_random_flips(self, number_of_flips):
    self.c_ising.get().try_many_random_flips(number_of_flips)
