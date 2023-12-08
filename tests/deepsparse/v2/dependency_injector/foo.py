class Foo:
    # @inject
    # def __init__(self, my_dependency: MyDependency):
    #     self.my_dependency = my_dependency
    
    def __init__(self):
        ...

    @inject
    def bar(self,  my_dependency: MyDependenc):
        print(f"Value from MyDependency: {my_dependency.value}")
