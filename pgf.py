#provides a class for dumping tab separated files for PGF


class PGF:
	def __init__(self):
		self.column_names = []
		self.columns = []
	def add(self, name, column):
		if len(self.columns) > 1:
			assert len(self.columns[0]) == len(column)

		self.columns.append(column)
		self.column_names.append(name)


	def write(self, filename):
		f = open(filename,'w')

		for name in self.column_names:
			f.write(name + '\t')
		f.write("\n")		

		for j in range(len(self.columns[0])):
			for col in self.columns:
				f.write("{}\t".format(float(col[j])))
			f.write("\n")

		f.close()
