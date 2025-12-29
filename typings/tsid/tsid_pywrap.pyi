from __future__ import annotations
import typing
__all__: list[str] = ['AngleAxis', 'ConstraintBound', 'ConstraintEquality', 'ConstraintInequality', 'ConstraintLevel', 'Contact6d', 'ContactPoint', 'ContactTwoFramePositions', 'Exception', 'FIXED_BASE_SYSTEM', 'FLOATING_BASE_SYSTEM', 'HQPData', 'HQPOutput', 'InverseDynamicsFormulationAccForce', 'Measured6dWrench', 'QPData', 'QPDataBase', 'QPDataQuadProg', 'Quaternion', 'RobotWrapper', 'RootJointType', 'SE3ToVector', 'SolverHQuadProg', 'SolverHQuadProgFast', 'TaskAMEquality', 'TaskActuationBounds', 'TaskActuationEquality', 'TaskComEquality', 'TaskCopEquality', 'TaskJointBounds', 'TaskJointPosVelAccBounds', 'TaskJointPosture', 'TaskSE3Equality', 'TaskTwoFramesEquality', 'TrajectoryEuclidianConstant', 'TrajectorySE3Constant', 'TrajectorySample', 'boost_type_index', 'seed', 'sharedMemory', 'std_type_index', 'vectorToSE3']
class AngleAxis(Boost.Python.instance):
    """
    AngleAxis representation of a rotation.
    
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (AngleAxis)arg1, (AngleAxis)arg2) -> bool :
        
            C++ signature :
                bool __eq__(Eigen::AngleAxis<double>,Eigen::AngleAxis<double>)
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Default constructor
        
            C++ signature :
                void __init__(_object*)
        
        __init__( (object)self, (float)angle, (numpy.ndarray)axis) -> None :
            Initialize from angle and axis.
        
            C++ signature :
                void __init__(_object*,double,Eigen::Matrix<double, 3, 1, 0, 3, 1>)
        
        __init__( (object)self, (numpy.ndarray)R) -> None :
            Initialize from a rotation matrix
        
            C++ signature :
                void __init__(_object*,Eigen::Matrix<double, 3, 3, 0, 3, 3>)
        
        __init__( (object)self, (Quaternion)quaternion) -> None :
            Initialize from a quaternion.
        
            C++ signature :
                void __init__(_object*,Eigen::Quaternion<double, 0>)
        
        __init__( (object)self, (AngleAxis)copy) -> None :
            Copy constructor.
        
            C++ signature :
                void __init__(_object*,Eigen::AngleAxis<double>)
        """
    @staticmethod
    def __mul__(*args, **kwargs):
        """
        
        __mul__( (AngleAxis)arg1, (numpy.ndarray)arg2) -> object :
        
            C++ signature :
                _object* __mul__(Eigen::AngleAxis<double> {lvalue},Eigen::Matrix<double, 3, 1, 0, 3, 1>)
        
        __mul__( (AngleAxis)arg1, (Quaternion)arg2) -> object :
        
            C++ signature :
                _object* __mul__(Eigen::AngleAxis<double> {lvalue},Eigen::Quaternion<double, 0>)
        
        __mul__( (AngleAxis)arg1, (AngleAxis)arg2) -> object :
        
            C++ signature :
                _object* __mul__(Eigen::AngleAxis<double> {lvalue},Eigen::AngleAxis<double>)
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (AngleAxis)arg1, (AngleAxis)arg2) -> bool :
        
            C++ signature :
                bool __ne__(Eigen::AngleAxis<double>,Eigen::AngleAxis<double>)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (AngleAxis)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > __repr__(Eigen::AngleAxis<double>)
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (AngleAxis)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > __str__(Eigen::AngleAxis<double>)
        """
    @staticmethod
    def fromRotationMatrix(*args, **kwargs):
        """
        
        fromRotationMatrix( (AngleAxis)self, (numpy.ndarray)rotation matrix) -> AngleAxis :
            Sets *this from a 3x3 rotation matrix
        
            C++ signature :
                Eigen::AngleAxis<double> {lvalue} fromRotationMatrix(Eigen::AngleAxis<double> {lvalue},Eigen::MatrixBase<Eigen::Matrix<double, 3, 3, 0, 3, 3> >)
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (AngleAxis)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        
            C++ signature :
                long id(Eigen::AngleAxis<double>)
        """
    @staticmethod
    def inverse(*args, **kwargs):
        """
        
        inverse( (AngleAxis)self) -> AngleAxis :
            Return the inverse rotation.
        
            C++ signature :
                Eigen::AngleAxis<double> inverse(Eigen::AngleAxis<double> {lvalue})
        """
    @staticmethod
    def isApprox(*args, **kwargs):
        """
        
        isApprox( (AngleAxis)self, (AngleAxis)other [, (float)prec]) -> bool :
            Returns true if *this is approximately equal to other, within the precision determined by prec.
        
            C++ signature :
                bool isApprox(Eigen::AngleAxis<double>,Eigen::AngleAxis<double> [,double])
        """
    @staticmethod
    def matrix(*args, **kwargs):
        """
        
        matrix( (AngleAxis)self) -> numpy.ndarray :
            Returns an equivalent rotation matrix.
        
            C++ signature :
                Eigen::Matrix<double, 3, 3, 0, 3, 3> matrix(Eigen::AngleAxis<double> {lvalue})
        """
    @staticmethod
    def toRotationMatrix(*args, **kwargs):
        """
        
        toRotationMatrix( (AngleAxis)arg1) -> numpy.ndarray :
            Constructs and returns an equivalent rotation matrix.
        
            C++ signature :
                Eigen::Matrix<double, 3, 3, 0, 3, 3> toRotationMatrix(Eigen::AngleAxis<double> {lvalue})
        """
    @property
    def angle(*args, **kwargs):
        """
        The rotation angle.
        """
    @angle.setter
    def angle(*args, **kwargs):
        ...
    @property
    def axis(*args, **kwargs):
        """
        The rotation axis.
        """
    @axis.setter
    def axis(*args, **kwargs):
        ...
class ConstraintBound(Boost.Python.instance):
    """
    Constraint Bound info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name) -> None :
            Default constructor with name.
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
        
        __init__( (object)arg1, (str)name, (int)size) -> None :
            Default constructor with name and size.
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,unsigned int)
        
        __init__( (object)arg1, (str)name, (numpy.ndarray)lb, (numpy.ndarray)ub) -> None :
            Default constructor with name and constraint.
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def resize(*args, **kwargs):
        """
        
        resize( (ConstraintBound)arg1, (int)r, (int)c) -> None :
            Resize constraint size.
        
            C++ signature :
                void resize(tsid::math::ConstraintBound {lvalue},unsigned int,unsigned int)
        """
    @staticmethod
    def setLowerBound(*args, **kwargs):
        """
        
        setLowerBound( (ConstraintBound)arg1, (numpy.ndarray)lb) -> bool :
            Set LowerBound
        
            C++ signature :
                bool setLowerBound(tsid::math::ConstraintBound {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >)
        """
    @staticmethod
    def setUpperBound(*args, **kwargs):
        """
        
        setUpperBound( (ConstraintBound)arg1, (numpy.ndarray)ub) -> bool :
            Set UpperBound
        
            C++ signature :
                bool setUpperBound(tsid::math::ConstraintBound {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >)
        """
    @staticmethod
    def setVector(*args, **kwargs):
        """
        
        setVector( (ConstraintBound)arg1, (numpy.ndarray)vector) -> bool :
            Set Vector
        
            C++ signature :
                bool setVector(tsid::math::ConstraintBound {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >)
        """
    @property
    def cols(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintBound)arg1) -> int :
        
            C++ signature :
                unsigned int None(tsid::math::ConstraintBound {lvalue})
        """
    @property
    def isBound(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintBound)arg1) -> bool :
        
            C++ signature :
                bool None(tsid::math::ConstraintBound {lvalue})
        """
    @property
    def isEquality(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintBound)arg1) -> bool :
        
            C++ signature :
                bool None(tsid::math::ConstraintBound {lvalue})
        """
    @property
    def isInequality(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintBound)arg1) -> bool :
        
            C++ signature :
                bool None(tsid::math::ConstraintBound {lvalue})
        """
    @property
    def lowerBound(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintBound)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::math::ConstraintBound)
        """
    @property
    def rows(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintBound)arg1) -> int :
        
            C++ signature :
                unsigned int None(tsid::math::ConstraintBound {lvalue})
        """
    @property
    def upperBound(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintBound)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::math::ConstraintBound)
        """
    @property
    def vector(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintBound)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::math::ConstraintBound)
        """
class ConstraintEquality(Boost.Python.instance):
    """
    Constraint Equality info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name) -> None :
            Default constructor with name.
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
        
        __init__( (object)arg1, (str)name, (int)row, (int)col) -> None :
            Default constructor with name and size.
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,unsigned int,unsigned int)
        
        __init__( (object)arg1, (str)name, (numpy.ndarray)A, (numpy.ndarray)b) -> None :
            Default constructor with name and constraint.
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,Eigen::Matrix<double, -1, -1, 0, -1, -1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def resize(*args, **kwargs):
        """
        
        resize( (ConstraintEquality)arg1, (int)r, (int)c) -> None :
            Resize constraint size.
        
            C++ signature :
                void resize(tsid::math::ConstraintEquality {lvalue},unsigned int,unsigned int)
        """
    @staticmethod
    def setLowerBound(*args, **kwargs):
        """
        
        setLowerBound( (ConstraintEquality)arg1, (numpy.ndarray)lb) -> bool :
            Set LowerBound
        
            C++ signature :
                bool setLowerBound(tsid::math::ConstraintEquality {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >)
        """
    @staticmethod
    def setMatrix(*args, **kwargs):
        """
        
        setMatrix( (ConstraintEquality)arg1, (numpy.ndarray)matrix) -> bool :
            Set Matrix
        
            C++ signature :
                bool setMatrix(tsid::math::ConstraintEquality {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, -1, 0, -1, -1> const, 0, Eigen::OuterStride<-1> >)
        """
    @staticmethod
    def setUpperBound(*args, **kwargs):
        """
        
        setUpperBound( (ConstraintEquality)arg1, (numpy.ndarray)ub) -> bool :
            Set UpperBound
        
            C++ signature :
                bool setUpperBound(tsid::math::ConstraintEquality {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >)
        """
    @staticmethod
    def setVector(*args, **kwargs):
        """
        
        setVector( (ConstraintEquality)arg1, (numpy.ndarray)vector) -> bool :
            Set Vector
        
            C++ signature :
                bool setVector(tsid::math::ConstraintEquality {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >)
        """
    @property
    def cols(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintEquality)arg1) -> int :
        
            C++ signature :
                unsigned int None(tsid::math::ConstraintEquality {lvalue})
        """
    @property
    def isBound(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintEquality)arg1) -> bool :
        
            C++ signature :
                bool None(tsid::math::ConstraintEquality {lvalue})
        """
    @property
    def isEquality(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintEquality)arg1) -> bool :
        
            C++ signature :
                bool None(tsid::math::ConstraintEquality {lvalue})
        """
    @property
    def isInequality(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintEquality)arg1) -> bool :
        
            C++ signature :
                bool None(tsid::math::ConstraintEquality {lvalue})
        """
    @property
    def lowerBound(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::math::ConstraintEquality)
        """
    @property
    def matrix(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, -1, 0, -1, -1> None(tsid::math::ConstraintEquality)
        """
    @property
    def rows(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintEquality)arg1) -> int :
        
            C++ signature :
                unsigned int None(tsid::math::ConstraintEquality {lvalue})
        """
    @property
    def upperBound(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::math::ConstraintEquality)
        """
    @property
    def vector(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::math::ConstraintEquality)
        """
class ConstraintInequality(Boost.Python.instance):
    """
    Constraint Inequality info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name) -> None :
            Default constructor with name.
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
        
        __init__( (object)arg1, (str)name, (int)row, (int)col) -> None :
            Default constructor with name and size.
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,unsigned int,unsigned int)
        
        __init__( (object)arg1, (str)name, (numpy.ndarray)A, (numpy.ndarray)lb, (numpy.ndarray)ub) -> None :
            Default constructor with name and constraint.
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,Eigen::Matrix<double, -1, -1, 0, -1, -1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def resize(*args, **kwargs):
        """
        
        resize( (ConstraintInequality)arg1, (int)r, (int)c) -> None :
            Resize constraint size.
        
            C++ signature :
                void resize(tsid::math::ConstraintInequality {lvalue},unsigned int,unsigned int)
        """
    @staticmethod
    def setLowerBound(*args, **kwargs):
        """
        
        setLowerBound( (ConstraintInequality)arg1, (numpy.ndarray)lb) -> bool :
            Set LowerBound
        
            C++ signature :
                bool setLowerBound(tsid::math::ConstraintInequality {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >)
        """
    @staticmethod
    def setMatrix(*args, **kwargs):
        """
        
        setMatrix( (ConstraintInequality)arg1, (numpy.ndarray)matrix) -> bool :
            Set Matrix
        
            C++ signature :
                bool setMatrix(tsid::math::ConstraintInequality {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, -1, 0, -1, -1> const, 0, Eigen::OuterStride<-1> >)
        """
    @staticmethod
    def setUpperBound(*args, **kwargs):
        """
        
        setUpperBound( (ConstraintInequality)arg1, (numpy.ndarray)ub) -> bool :
            Set UpperBound
        
            C++ signature :
                bool setUpperBound(tsid::math::ConstraintInequality {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >)
        """
    @staticmethod
    def setVector(*args, **kwargs):
        """
        
        setVector( (ConstraintInequality)arg1, (numpy.ndarray)vector) -> bool :
            Set Vector
        
            C++ signature :
                bool setVector(tsid::math::ConstraintInequality {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >)
        """
    @property
    def cols(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintInequality)arg1) -> int :
        
            C++ signature :
                unsigned int None(tsid::math::ConstraintInequality {lvalue})
        """
    @property
    def isBound(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintInequality)arg1) -> bool :
        
            C++ signature :
                bool None(tsid::math::ConstraintInequality {lvalue})
        """
    @property
    def isEquality(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintInequality)arg1) -> bool :
        
            C++ signature :
                bool None(tsid::math::ConstraintInequality {lvalue})
        """
    @property
    def isInequality(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintInequality)arg1) -> bool :
        
            C++ signature :
                bool None(tsid::math::ConstraintInequality {lvalue})
        """
    @property
    def lowerBound(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintInequality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::math::ConstraintInequality)
        """
    @property
    def matrix(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintInequality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, -1, 0, -1, -1> None(tsid::math::ConstraintInequality)
        """
    @property
    def rows(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintInequality)arg1) -> int :
        
            C++ signature :
                unsigned int None(tsid::math::ConstraintInequality {lvalue})
        """
    @property
    def upperBound(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintInequality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::math::ConstraintInequality)
        """
    @property
    def vector(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ConstraintInequality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::math::ConstraintInequality)
        """
class ConstraintLevel(Boost.Python.instance):
    """
    ConstraintLevel info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (ConstraintLevel)arg1, (float)arg2, (ConstraintEquality)data) -> None :
        
            C++ signature :
                void append(tsid::python::ConstraintLevels {lvalue},double,std::shared_ptr<tsid::math::ConstraintEquality>)
        
        append( (ConstraintLevel)arg1, (float)arg2, (ConstraintInequality)data) -> None :
        
            C++ signature :
                void append(tsid::python::ConstraintLevels {lvalue},double,std::shared_ptr<tsid::math::ConstraintInequality>)
        
        append( (ConstraintLevel)arg1, (float)arg2, (ConstraintBound)data) -> None :
        
            C++ signature :
                void append(tsid::python::ConstraintLevels {lvalue},double,std::shared_ptr<tsid::math::ConstraintBound>)
        """
    @staticmethod
    def print_all(*args, **kwargs):
        """
        
        print_all( (ConstraintLevel)arg1) -> None :
        
            C++ signature :
                void print_all(tsid::python::ConstraintLevels {lvalue})
        """
class Contact6d(Boost.Python.instance):
    """
    Contact6d info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot, (str)framename, (numpy.ndarray)contactPoint, (numpy.ndarray)contactNormal, (float)frictionCoeff, (float)minForce, (float)maxForce) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,Eigen::Matrix<double, -1, -1, 0, -1, -1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,double,double,double)
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot, (str)framename, (numpy.ndarray)contactPoint, (numpy.ndarray)contactNormal, (float)frictionCoeff, (float)minForce, (float)maxForce, (float)wForceReg) -> None :
            Deprecated Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,Eigen::Matrix<double, -1, -1, 0, -1, -1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,double,double,double,double)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def computeForceRegularizationTask(*args, **kwargs):
        """
        
        computeForceRegularizationTask( (Contact6d)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality computeForceRegularizationTask(tsid::contacts::Contact6d {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>)
        """
    @staticmethod
    def computeForceTask(*args, **kwargs):
        """
        
        computeForceTask( (Contact6d)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintInequality :
        
            C++ signature :
                tsid::math::ConstraintInequality computeForceTask(tsid::contacts::Contact6d {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>)
        """
    @staticmethod
    def computeMotionTask(*args, **kwargs):
        """
        
        computeMotionTask( (Contact6d)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality computeMotionTask(tsid::contacts::Contact6d {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def getMotionTask(*args, **kwargs):
        """
        
        getMotionTask( (Contact6d)arg1) -> TaskSE3Equality :
        
            C++ signature :
                tsid::tasks::TaskSE3Equality getMotionTask(tsid::contacts::Contact6d {lvalue})
        """
    @staticmethod
    def getNormalForce(*args, **kwargs):
        """
        
        getNormalForce( (Contact6d)arg1, (numpy.ndarray)vec) -> float :
        
            C++ signature :
                double getNormalForce(tsid::contacts::Contact6d {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setContactNormal(*args, **kwargs):
        """
        
        setContactNormal( (Contact6d)arg1, (numpy.ndarray)vec) -> bool :
        
            C++ signature :
                bool setContactNormal(tsid::contacts::Contact6d {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setContactPoints(*args, **kwargs):
        """
        
        setContactPoints( (Contact6d)arg1, (numpy.ndarray)vec) -> bool :
        
            C++ signature :
                bool setContactPoints(tsid::contacts::Contact6d {lvalue},Eigen::Matrix<double, -1, -1, 0, -1, -1>)
        """
    @staticmethod
    def setForceReference(*args, **kwargs):
        """
        
        setForceReference( (Contact6d)arg1, (numpy.ndarray)f_vec) -> None :
        
            C++ signature :
                void setForceReference(tsid::contacts::Contact6d {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setFrictionCoefficient(*args, **kwargs):
        """
        
        setFrictionCoefficient( (Contact6d)arg1, (float)friction_coeff) -> bool :
        
            C++ signature :
                bool setFrictionCoefficient(tsid::contacts::Contact6d {lvalue},double)
        """
    @staticmethod
    def setKd(*args, **kwargs):
        """
        
        setKd( (Contact6d)arg1, (numpy.ndarray)Kd) -> None :
        
            C++ signature :
                void setKd(tsid::contacts::Contact6d {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setKp(*args, **kwargs):
        """
        
        setKp( (Contact6d)arg1, (numpy.ndarray)Kp) -> None :
        
            C++ signature :
                void setKp(tsid::contacts::Contact6d {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setMaxNormalForce(*args, **kwargs):
        """
        
        setMaxNormalForce( (Contact6d)arg1, (float)max_force) -> bool :
        
            C++ signature :
                bool setMaxNormalForce(tsid::contacts::Contact6d {lvalue},double)
        """
    @staticmethod
    def setMinNormalForce(*args, **kwargs):
        """
        
        setMinNormalForce( (Contact6d)arg1, (float)min_force) -> bool :
        
            C++ signature :
                bool setMinNormalForce(tsid::contacts::Contact6d {lvalue},double)
        """
    @staticmethod
    def setReference(*args, **kwargs):
        """
        
        setReference( (Contact6d)arg1, (object)SE3) -> None :
        
            C++ signature :
                void setReference(tsid::contacts::Contact6d {lvalue},pinocchio::SE3Tpl<double, 0>)
        """
    @staticmethod
    def setRegularizationTaskWeightVector(*args, **kwargs):
        """
        
        setRegularizationTaskWeightVector( (Contact6d)arg1, (numpy.ndarray)w_vec) -> None :
        
            C++ signature :
                void setRegularizationTaskWeightVector(tsid::contacts::Contact6d {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @property
    def Kd(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.Contact6d)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::contacts::Contact6d {lvalue})
        """
    @property
    def Kp(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.Contact6d)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::contacts::Contact6d {lvalue})
        """
    @property
    def getForceGeneratorMatrix(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.Contact6d)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, -1, 0, -1, -1> None(tsid::contacts::Contact6d {lvalue})
        """
    @property
    def getMaxNormalForce(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.Contact6d)arg1) -> float :
        
            C++ signature :
                double None(tsid::contacts::Contact6d {lvalue})
        """
    @property
    def getMinNormalForce(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.Contact6d)arg1) -> float :
        
            C++ signature :
                double None(tsid::contacts::Contact6d {lvalue})
        """
    @property
    def n_force(*args, **kwargs):
        """
        return number of force
        """
    @property
    def n_motion(*args, **kwargs):
        """
        return number of motion
        """
    @property
    def name(*args, **kwargs):
        """
        return name
        """
class ContactPoint(Boost.Python.instance):
    """
    ContactPoint info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot, (str)framename, (numpy.ndarray)contactNormal, (float)frictionCoeff, (float)minForce, (float)maxForce) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,Eigen::Matrix<double, -1, 1, 0, -1, 1>,double,double,double)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def computeForceRegularizationTask(*args, **kwargs):
        """
        
        computeForceRegularizationTask( (ContactPoint)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality computeForceRegularizationTask(tsid::contacts::ContactPoint {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>)
        """
    @staticmethod
    def computeForceTask(*args, **kwargs):
        """
        
        computeForceTask( (ContactPoint)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintInequality :
        
            C++ signature :
                tsid::math::ConstraintInequality computeForceTask(tsid::contacts::ContactPoint {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>)
        """
    @staticmethod
    def computeMotionTask(*args, **kwargs):
        """
        
        computeMotionTask( (ContactPoint)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality computeMotionTask(tsid::contacts::ContactPoint {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def getNormalForce(*args, **kwargs):
        """
        
        getNormalForce( (ContactPoint)arg1, (numpy.ndarray)vec) -> float :
        
            C++ signature :
                double getNormalForce(tsid::contacts::ContactPoint {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setContactNormal(*args, **kwargs):
        """
        
        setContactNormal( (ContactPoint)arg1, (numpy.ndarray)vec) -> bool :
        
            C++ signature :
                bool setContactNormal(tsid::contacts::ContactPoint {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setForceReference(*args, **kwargs):
        """
        
        setForceReference( (ContactPoint)arg1, (numpy.ndarray)f_vec) -> None :
        
            C++ signature :
                void setForceReference(tsid::contacts::ContactPoint {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setFrictionCoefficient(*args, **kwargs):
        """
        
        setFrictionCoefficient( (ContactPoint)arg1, (float)friction_coeff) -> bool :
        
            C++ signature :
                bool setFrictionCoefficient(tsid::contacts::ContactPoint {lvalue},double)
        """
    @staticmethod
    def setKd(*args, **kwargs):
        """
        
        setKd( (ContactPoint)arg1, (numpy.ndarray)Kd) -> None :
        
            C++ signature :
                void setKd(tsid::contacts::ContactPoint {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setKp(*args, **kwargs):
        """
        
        setKp( (ContactPoint)arg1, (numpy.ndarray)Kp) -> None :
        
            C++ signature :
                void setKp(tsid::contacts::ContactPoint {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setMaxNormalForce(*args, **kwargs):
        """
        
        setMaxNormalForce( (ContactPoint)arg1, (float)max_force) -> bool :
        
            C++ signature :
                bool setMaxNormalForce(tsid::contacts::ContactPoint {lvalue},double)
        """
    @staticmethod
    def setMinNormalForce(*args, **kwargs):
        """
        
        setMinNormalForce( (ContactPoint)arg1, (float)min_force) -> bool :
        
            C++ signature :
                bool setMinNormalForce(tsid::contacts::ContactPoint {lvalue},double)
        """
    @staticmethod
    def setReference(*args, **kwargs):
        """
        
        setReference( (ContactPoint)arg1, (object)SE3) -> None :
        
            C++ signature :
                void setReference(tsid::contacts::ContactPoint {lvalue},pinocchio::SE3Tpl<double, 0>)
        """
    @staticmethod
    def setRegularizationTaskWeightVector(*args, **kwargs):
        """
        
        setRegularizationTaskWeightVector( (ContactPoint)arg1, (numpy.ndarray)w_vec) -> None :
        
            C++ signature :
                void setRegularizationTaskWeightVector(tsid::contacts::ContactPoint {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def useLocalFrame(*args, **kwargs):
        """
        
        useLocalFrame( (ContactPoint)arg1, (bool)local_frame) -> None :
        
            C++ signature :
                void useLocalFrame(tsid::contacts::ContactPoint {lvalue},bool)
        """
    @property
    def Kd(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ContactPoint)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::contacts::ContactPoint {lvalue})
        """
    @property
    def Kp(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ContactPoint)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::contacts::ContactPoint {lvalue})
        """
    @property
    def getForceGeneratorMatrix(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ContactPoint)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, -1, 0, -1, -1> None(tsid::contacts::ContactPoint {lvalue})
        """
    @property
    def getMaxNormalForce(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ContactPoint)arg1) -> float :
        
            C++ signature :
                double None(tsid::contacts::ContactPoint {lvalue})
        """
    @property
    def getMinNormalForce(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ContactPoint)arg1) -> float :
        
            C++ signature :
                double None(tsid::contacts::ContactPoint {lvalue})
        """
    @property
    def n_force(*args, **kwargs):
        """
        return number of force
        """
    @property
    def n_motion(*args, **kwargs):
        """
        return number of motion
        """
    @property
    def name(*args, **kwargs):
        """
        return name
        """
class ContactTwoFramePositions(Boost.Python.instance):
    """
    ContactTwoFramePositions info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot, (str)framename1, (str)framename2, (float)minForce, (float)maxForce) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,double,double)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def computeForceRegularizationTask(*args, **kwargs):
        """
        
        computeForceRegularizationTask( (ContactTwoFramePositions)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality computeForceRegularizationTask(tsid::contacts::ContactTwoFramePositions {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>)
        """
    @staticmethod
    def computeForceTask(*args, **kwargs):
        """
        
        computeForceTask( (ContactTwoFramePositions)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintInequality :
        
            C++ signature :
                tsid::math::ConstraintInequality computeForceTask(tsid::contacts::ContactTwoFramePositions {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>)
        """
    @staticmethod
    def computeMotionTask(*args, **kwargs):
        """
        
        computeMotionTask( (ContactTwoFramePositions)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality computeMotionTask(tsid::contacts::ContactTwoFramePositions {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def getNormalForce(*args, **kwargs):
        """
        
        getNormalForce( (ContactTwoFramePositions)arg1, (numpy.ndarray)vec) -> float :
        
            C++ signature :
                double getNormalForce(tsid::contacts::ContactTwoFramePositions {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setContactNormal(*args, **kwargs):
        """
        
        setContactNormal( (ContactTwoFramePositions)arg1, (numpy.ndarray)vec) -> bool :
        
            C++ signature :
                bool setContactNormal(tsid::contacts::ContactTwoFramePositions {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setForceReference(*args, **kwargs):
        """
        
        setForceReference( (ContactTwoFramePositions)arg1, (numpy.ndarray)f_vec) -> None :
        
            C++ signature :
                void setForceReference(tsid::contacts::ContactTwoFramePositions {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setFrictionCoefficient(*args, **kwargs):
        """
        
        setFrictionCoefficient( (ContactTwoFramePositions)arg1, (float)friction_coeff) -> bool :
        
            C++ signature :
                bool setFrictionCoefficient(tsid::contacts::ContactTwoFramePositions {lvalue},double)
        """
    @staticmethod
    def setKd(*args, **kwargs):
        """
        
        setKd( (ContactTwoFramePositions)arg1, (numpy.ndarray)Kd) -> None :
        
            C++ signature :
                void setKd(tsid::contacts::ContactTwoFramePositions {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setKp(*args, **kwargs):
        """
        
        setKp( (ContactTwoFramePositions)arg1, (numpy.ndarray)Kp) -> None :
        
            C++ signature :
                void setKp(tsid::contacts::ContactTwoFramePositions {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setMaxNormalForce(*args, **kwargs):
        """
        
        setMaxNormalForce( (ContactTwoFramePositions)arg1, (float)max_force) -> bool :
        
            C++ signature :
                bool setMaxNormalForce(tsid::contacts::ContactTwoFramePositions {lvalue},double)
        """
    @staticmethod
    def setMinNormalForce(*args, **kwargs):
        """
        
        setMinNormalForce( (ContactTwoFramePositions)arg1, (float)min_force) -> bool :
        
            C++ signature :
                bool setMinNormalForce(tsid::contacts::ContactTwoFramePositions {lvalue},double)
        """
    @staticmethod
    def setRegularizationTaskWeightVector(*args, **kwargs):
        """
        
        setRegularizationTaskWeightVector( (ContactTwoFramePositions)arg1, (numpy.ndarray)w_vec) -> None :
        
            C++ signature :
                void setRegularizationTaskWeightVector(tsid::contacts::ContactTwoFramePositions {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @property
    def Kd(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ContactTwoFramePositions)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::contacts::ContactTwoFramePositions {lvalue})
        """
    @property
    def Kp(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ContactTwoFramePositions)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::contacts::ContactTwoFramePositions {lvalue})
        """
    @property
    def getForceGeneratorMatrix(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ContactTwoFramePositions)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, -1, 0, -1, -1> None(tsid::contacts::ContactTwoFramePositions {lvalue})
        """
    @property
    def getMaxNormalForce(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ContactTwoFramePositions)arg1) -> float :
        
            C++ signature :
                double None(tsid::contacts::ContactTwoFramePositions {lvalue})
        """
    @property
    def getMinNormalForce(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.ContactTwoFramePositions)arg1) -> float :
        
            C++ signature :
                double None(tsid::contacts::ContactTwoFramePositions {lvalue})
        """
    @property
    def n_force(*args, **kwargs):
        """
        return number of force
        """
    @property
    def n_motion(*args, **kwargs):
        """
        return number of motion
        """
    @property
    def name(*args, **kwargs):
        """
        return name
        """
class Exception(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 72
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)arg2) -> None :
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @property
    def message(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.Exception)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > None(eigenpy::Exception {lvalue})
        """
class HQPData(Boost.Python.instance):
    """
    HQPdata info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (HQPData)arg1, (ConstraintLevel)constraintLevel) -> None :
        
            C++ signature :
                void append(tsid::python::HQPDatas {lvalue},tsid::python::ConstraintLevels*)
        """
    @staticmethod
    def print_all(*args, **kwargs):
        """
        
        print_all( (HQPData)arg1) -> None :
        
            C++ signature :
                void print_all(tsid::python::HQPDatas {lvalue})
        """
    @staticmethod
    def resize(*args, **kwargs):
        """
        
        resize( (HQPData)arg1, (int)i) -> None :
        
            C++ signature :
                void resize(tsid::python::HQPDatas {lvalue},unsigned long)
        """
class HQPOutput(Boost.Python.instance):
    """
    HQPOutput info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*)
        
        __init__( (object)arg1, (int)nVars, (int)nEq, (int)nInCon) -> None :
        
            C++ signature :
                void __init__(_object*,int,int,int)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @property
    def status(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.HQPOutput)arg1) -> int :
        
            C++ signature :
                int None(tsid::solvers::HQPOutput)
        """
    @property
    def x(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.HQPOutput)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::solvers::HQPOutput)
        """
class InverseDynamicsFormulationAccForce(Boost.Python.instance):
    """
    InvDyn info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot, (bool)verbose) -> None :
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue},bool)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def addActuationTask(*args, **kwargs):
        """
        
        addActuationTask( (InverseDynamicsFormulationAccForce)arg1, (TaskActuationBounds)task, (float)weight, (int)priorityLevel, (float)transition duration) -> bool :
        
            C++ signature :
                bool addActuationTask(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::tasks::TaskActuationBounds {lvalue},double,unsigned int,double)
        """
    @staticmethod
    def addForceTask(*args, **kwargs):
        """
        
        addForceTask( (InverseDynamicsFormulationAccForce)arg1, (TaskCopEquality)task, (float)weight, (int)priorityLevel, (float)transition duration) -> bool :
        
            C++ signature :
                bool addForceTask(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::tasks::TaskCopEquality {lvalue},double,unsigned int,double)
        """
    @staticmethod
    def addMeasuredForce(*args, **kwargs):
        """
        
        addMeasuredForce( (InverseDynamicsFormulationAccForce)arg1, (Measured6dWrench)measured_force) -> bool :
        
            C++ signature :
                bool addMeasuredForce(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::contacts::Measured6Dwrench {lvalue})
        """
    @staticmethod
    def addMotionTask(*args, **kwargs):
        """
        
        addMotionTask( (InverseDynamicsFormulationAccForce)arg1, (TaskSE3Equality)task, (float)weight, (int)priorityLevel, (float)transition duration) -> bool :
        
            C++ signature :
                bool addMotionTask(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::tasks::TaskSE3Equality {lvalue},double,unsigned int,double)
        
        addMotionTask( (InverseDynamicsFormulationAccForce)arg1, (TaskComEquality)task, (float)weight, (int)priorityLevel, (float)transition duration) -> bool :
        
            C++ signature :
                bool addMotionTask(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::tasks::TaskComEquality {lvalue},double,unsigned int,double)
        
        addMotionTask( (InverseDynamicsFormulationAccForce)arg1, (TaskJointPosture)task, (float)weight, (int)priorityLevel, (float)transition duration) -> bool :
        
            C++ signature :
                bool addMotionTask(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::tasks::TaskJointPosture {lvalue},double,unsigned int,double)
        
        addMotionTask( (InverseDynamicsFormulationAccForce)arg1, (TaskJointBounds)task, (float)weight, (int)priorityLevel, (float)transition duration) -> bool :
        
            C++ signature :
                bool addMotionTask(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::tasks::TaskJointBounds {lvalue},double,unsigned int,double)
        
        addMotionTask( (InverseDynamicsFormulationAccForce)arg1, (TaskJointPosVelAccBounds)task, (float)weight, (int)priorityLevel, (float)transition duration) -> bool :
        
            C++ signature :
                bool addMotionTask(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::tasks::TaskJointPosVelAccBounds {lvalue},double,unsigned int,double)
        
        addMotionTask( (InverseDynamicsFormulationAccForce)arg1, (TaskAMEquality)task, (float)weight, (int)priorityLevel, (float)transition duration) -> bool :
        
            C++ signature :
                bool addMotionTask(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::tasks::TaskAMEquality {lvalue},double,unsigned int,double)
        
        addMotionTask( (InverseDynamicsFormulationAccForce)arg1, (TaskTwoFramesEquality)task, (float)weight, (int)priorityLevel, (float)transition duration) -> bool :
        
            C++ signature :
                bool addMotionTask(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::tasks::TaskTwoFramesEquality {lvalue},double,unsigned int,double)
        """
    @staticmethod
    def addRigidContact(*args, **kwargs):
        """
        
        addRigidContact( (InverseDynamicsFormulationAccForce)arg1, (Contact6d)contact) -> bool :
        
            C++ signature :
                bool addRigidContact(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::contacts::Contact6d {lvalue})
        
        addRigidContact( (InverseDynamicsFormulationAccForce)arg1, (Contact6d)contact, (float)force_reg_weight) -> bool :
        
            C++ signature :
                bool addRigidContact(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::contacts::Contact6d {lvalue},double)
        
        addRigidContact( (InverseDynamicsFormulationAccForce)arg1, (Contact6d)contact, (float)force_reg_weight, (float)motion_weight, (bool)priority_level) -> bool :
        
            C++ signature :
                bool addRigidContact(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::contacts::Contact6d {lvalue},double,double,bool)
        
        addRigidContact( (InverseDynamicsFormulationAccForce)arg1, (ContactPoint)contact, (float)force_reg_weight) -> bool :
        
            C++ signature :
                bool addRigidContact(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::contacts::ContactPoint {lvalue},double)
        
        addRigidContact( (InverseDynamicsFormulationAccForce)arg1, (ContactPoint)contact, (float)force_reg_weight, (float)motion_weight, (bool)priority_level) -> bool :
        
            C++ signature :
                bool addRigidContact(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::contacts::ContactPoint {lvalue},double,double,bool)
        
        addRigidContact( (InverseDynamicsFormulationAccForce)arg1, (ContactTwoFramePositions)contact, (float)force_reg_weight) -> bool :
        
            C++ signature :
                bool addRigidContact(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::contacts::ContactTwoFramePositions {lvalue},double)
        
        addRigidContact( (InverseDynamicsFormulationAccForce)arg1, (ContactTwoFramePositions)contact, (float)force_reg_weight, (float)motion_weight, (bool)priority_level) -> bool :
        
            C++ signature :
                bool addRigidContact(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::contacts::ContactTwoFramePositions {lvalue},double,double,bool)
        """
    @staticmethod
    def checkContact(*args, **kwargs):
        """
        
        checkContact( (InverseDynamicsFormulationAccForce)arg1, (str)name, (HQPOutput)HQPOutput) -> bool :
        
            C++ signature :
                bool checkContact(tsid::InverseDynamicsFormulationAccForce {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::solvers::HQPOutput)
        """
    @staticmethod
    def computeProblemData(*args, **kwargs):
        """
        
        computeProblemData( (InverseDynamicsFormulationAccForce)arg1, (float)time, (numpy.ndarray)q, (numpy.ndarray)v) -> HQPData :
        
            C++ signature :
                tsid::python::HQPDatas computeProblemData(tsid::InverseDynamicsFormulationAccForce {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def data(*args, **kwargs):
        """
        
        data( (InverseDynamicsFormulationAccForce)arg1) -> object :
        
            C++ signature :
                pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> data(tsid::InverseDynamicsFormulationAccForce {lvalue})
        """
    @staticmethod
    def getAccelerations(*args, **kwargs):
        """
        
        getAccelerations( (InverseDynamicsFormulationAccForce)arg1, (HQPOutput)HQPOutput) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> getAccelerations(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::solvers::HQPOutput)
        """
    @staticmethod
    def getActuatorForces(*args, **kwargs):
        """
        
        getActuatorForces( (InverseDynamicsFormulationAccForce)arg1, (HQPOutput)HQPOutput) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> getActuatorForces(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::solvers::HQPOutput)
        """
    @staticmethod
    def getContactForce(*args, **kwargs):
        """
        
        getContactForce( (InverseDynamicsFormulationAccForce)arg1, (str)name, (HQPOutput)HQPOutput) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> getContactForce(tsid::InverseDynamicsFormulationAccForce {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::solvers::HQPOutput)
        """
    @staticmethod
    def getContactForces(*args, **kwargs):
        """
        
        getContactForces( (InverseDynamicsFormulationAccForce)arg1, (HQPOutput)HQPOutput) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> getContactForces(tsid::InverseDynamicsFormulationAccForce {lvalue},tsid::solvers::HQPOutput)
        """
    @staticmethod
    def removeFromHqpData(*args, **kwargs):
        """
        
        removeFromHqpData( (InverseDynamicsFormulationAccForce)arg1, (str)constraint_name) -> bool :
        
            C++ signature :
                bool removeFromHqpData(tsid::InverseDynamicsFormulationAccForce {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
        """
    @staticmethod
    def removeRigidContact(*args, **kwargs):
        """
        
        removeRigidContact( (InverseDynamicsFormulationAccForce)arg1, (str)contact_name, (float)duration) -> bool :
        
            C++ signature :
                bool removeRigidContact(tsid::InverseDynamicsFormulationAccForce {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,double)
        """
    @staticmethod
    def removeTask(*args, **kwargs):
        """
        
        removeTask( (InverseDynamicsFormulationAccForce)arg1, (str)task_name, (float)duration) -> bool :
        
            C++ signature :
                bool removeTask(tsid::InverseDynamicsFormulationAccForce {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,double)
        """
    @staticmethod
    def updateRigidContactWeights(*args, **kwargs):
        """
        
        updateRigidContactWeights( (InverseDynamicsFormulationAccForce)arg1, (str)contact_name, (float)force_regularization_weight) -> bool :
        
            C++ signature :
                bool updateRigidContactWeights(tsid::InverseDynamicsFormulationAccForce {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,double)
        
        updateRigidContactWeights( (InverseDynamicsFormulationAccForce)arg1, (str)contact_name, (float)force_regularization_weight, (float)motion_weight) -> bool :
        
            C++ signature :
                bool updateRigidContactWeights(tsid::InverseDynamicsFormulationAccForce {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,double,double)
        """
    @staticmethod
    def updateTaskWeight(*args, **kwargs):
        """
        
        updateTaskWeight( (InverseDynamicsFormulationAccForce)arg1, (str)task_name, (float)weight) -> bool :
        
            C++ signature :
                bool updateTaskWeight(tsid::InverseDynamicsFormulationAccForce {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,double)
        """
    @property
    def nEq(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.InverseDynamicsFormulationAccForce)arg1) -> int :
        
            C++ signature :
                unsigned int None(tsid::InverseDynamicsFormulationAccForce {lvalue})
        """
    @property
    def nIn(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.InverseDynamicsFormulationAccForce)arg1) -> int :
        
            C++ signature :
                unsigned int None(tsid::InverseDynamicsFormulationAccForce {lvalue})
        """
    @property
    def nVar(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.InverseDynamicsFormulationAccForce)arg1) -> int :
        
            C++ signature :
                unsigned int None(tsid::InverseDynamicsFormulationAccForce {lvalue})
        """
class Measured6dWrench(Boost.Python.instance):
    """
    Bindings for tsid::contacts::Measured6dwrench
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot, (str)frameName) -> None :
            Constructor for Measured6dwrench
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def computeJointTorques(*args, **kwargs):
        """
        
        computeJointTorques( (Measured6dWrench)arg1, (object)data) -> numpy.ndarray :
            Compute the joint torques from the measured contact force
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> computeJointTorques(tsid::contacts::Measured6Dwrench {lvalue},pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def setMeasuredContactForce(*args, **kwargs):
        """
        
        setMeasuredContactForce( (Measured6dWrench)arg1, (numpy.ndarray)fext) -> None :
            Set the measured contact force
        
            C++ signature :
                void setMeasuredContactForce(tsid::contacts::Measured6Dwrench {lvalue},Eigen::Matrix<double, 6, 1, 0, 6, 1>)
        """
    @staticmethod
    def useLocalFrame(*args, **kwargs):
        """
        
        useLocalFrame( (Measured6dWrench)arg1, (bool)local_frame) -> None :
            Specify whether to use the local frame for external force and jacobian
        
            C++ signature :
                void useLocalFrame(tsid::contacts::Measured6Dwrench {lvalue},bool)
        """
    @property
    def measuredContactForce(*args, **kwargs):
        """
        Get the measured contact force
        """
    @property
    def name(*args, **kwargs):
        """
        Get or set the name of the measured-6Dwrench instance
        """
    @name.setter
    def name(*args, **kwargs):
        ...
class QPData(QPDataBase):
    __instance_size__: typing.ClassVar[int] = 168
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None :
        
            C++ signature :
                void __init__(_object*)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @property
    def CI(*args, **kwargs):
        """
        Inequality constraint matrix
        """
    @property
    def lb(*args, **kwargs):
        """
        Inequality constraint lower bound
        """
    @property
    def ub(*args, **kwargs):
        """
        Inequality constraint upper bound
        """
class QPDataBase(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 104
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None :
        
            C++ signature :
                void __init__(_object*)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @property
    def CE(*args, **kwargs):
        """
        Equality constraint matrix
        """
    @property
    def H(*args, **kwargs):
        """
        Cost matrix
        """
    @property
    def ce0(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.QPDataBase)arg1) -> object :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> {lvalue} None(tsid::solvers::QPDataBaseTpl<double> {lvalue})
        """
    @property
    def g(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.QPDataBase)arg1) -> object :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> {lvalue} None(tsid::solvers::QPDataBaseTpl<double> {lvalue})
        """
class QPDataQuadProg(QPDataBase):
    __instance_size__: typing.ClassVar[int] = 152
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None :
        
            C++ signature :
                void __init__(_object*)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @property
    def CI(*args, **kwargs):
        """
        Inequality constraint matrix (unilateral)
        """
    @property
    def ci0(*args, **kwargs):
        """
        Inequality constraint vector (stacked lower and upper bounds)
        """
class Quaternion(Boost.Python.instance):
    """
    Quaternion representing rotation.
    
    Supported operations ('q is a Quaternion, 'v' is a Vector3): 'q*q' (rotation composition), 'q*=q', 'q*v' (rotating 'v' by 'q'), 'q==q', 'q!=q', 'q[0..3]'.
    """
    @staticmethod
    def FromTwoVectors(*args, **kwargs):
        """
        
        FromTwoVectors( (numpy.ndarray)a, (numpy.ndarray)b) -> Quaternion :
            Returns the quaternion which transforms a into b through a rotation.
        
            C++ signature :
                Eigen::Quaternion<double, 0>* FromTwoVectors(Eigen::Ref<Eigen::Matrix<double, 3, 1, 0, 3, 1> const, 0, Eigen::InnerStride<1> >,Eigen::Ref<Eigen::Matrix<double, 3, 1, 0, 3, 1> const, 0, Eigen::InnerStride<1> >)
        """
    @staticmethod
    def Identity(*args, **kwargs):
        """
        
        Identity() -> Quaternion :
            Returns a quaternion representing an identity rotation.
        
            C++ signature :
                Eigen::Quaternion<double, 0>* Identity()
        """
    @staticmethod
    def __abs__(*args, **kwargs):
        """
        
        __abs__( (Quaternion)arg1) -> float :
        
            C++ signature :
                double __abs__(Eigen::Quaternion<double, 0> {lvalue})
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (Quaternion)arg1, (Quaternion)arg2) -> bool :
        
            C++ signature :
                bool __eq__(Eigen::Quaternion<double, 0>,Eigen::Quaternion<double, 0>)
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (Quaternion)arg1, (int)arg2) -> float :
        
            C++ signature :
                double __getitem__(Eigen::Quaternion<double, 0>,int)
        """
    @staticmethod
    def __imul__(*args, **kwargs):
        """
        
        __imul__( (Quaternion)arg1, (Quaternion)arg2) -> object :
        
            C++ signature :
                _object* __imul__(boost::python::back_reference<Eigen::Quaternion<double, 0>&>,Eigen::Quaternion<double, 0>)
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (numpy.ndarray)R) -> object :
            Initialize from rotation matrix.
            	R : a rotation matrix 3x3.
        
            C++ signature :
                void* __init__(boost::python::api::object,Eigen::Ref<Eigen::Matrix<double, 3, 3, 0, 3, 3> const, 0, Eigen::OuterStride<-1> >)
        
        __init__( (object)arg1, (AngleAxis)aa) -> object :
            Initialize from an angle axis.
            	aa: angle axis object.
        
            C++ signature :
                void* __init__(boost::python::api::object,Eigen::AngleAxis<double>)
        
        __init__( (object)arg1, (Quaternion)quat) -> object :
            Copy constructor.
            	quat: a quaternion.
        
            C++ signature :
                void* __init__(boost::python::api::object,Eigen::Quaternion<double, 0>)
        
        __init__( (object)arg1, (numpy.ndarray)u, (numpy.ndarray)v) -> object :
            Initialize from two vectors u and v
        
            C++ signature :
                void* __init__(boost::python::api::object,Eigen::Ref<Eigen::Matrix<double, 3, 1, 0, 3, 1> const, 0, Eigen::InnerStride<1> >,Eigen::Ref<Eigen::Matrix<double, 3, 1, 0, 3, 1> const, 0, Eigen::InnerStride<1> >)
        
        __init__( (object)arg1, (numpy.ndarray)vec4) -> object :
            Initialize from a vector 4D.
            	vec4 : a 4D vector representing quaternion coefficients in the order xyzw.
        
            C++ signature :
                void* __init__(boost::python::api::object,Eigen::Ref<Eigen::Matrix<double, 4, 1, 0, 4, 1> const, 0, Eigen::InnerStride<1> >)
        
        __init__( (object)arg1) -> object :
            Default constructor
        
            C++ signature :
                void* __init__(boost::python::api::object)
        
        __init__( (object)arg1, (float)w, (float)x, (float)y, (float)z) -> object :
            Initialize from coefficients.
            
            ... note:: The order of coefficients is *w*, *x*, *y*, *z*. The [] operator numbers them differently, 0...4 for *x* *y* *z* *w*!
        
            C++ signature :
                void* __init__(boost::python::api::object,double,double,double,double)
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__() -> int :
        
            C++ signature :
                int __len__()
        """
    @staticmethod
    def __mul__(*args, **kwargs):
        """
        
        __mul__( (Quaternion)arg1, (Quaternion)arg2) -> object :
        
            C++ signature :
                _object* __mul__(Eigen::Quaternion<double, 0> {lvalue},Eigen::Quaternion<double, 0>)
        
        __mul__( (Quaternion)arg1, (numpy.ndarray)arg2) -> object :
        
            C++ signature :
                _object* __mul__(Eigen::Quaternion<double, 0> {lvalue},Eigen::Matrix<double, 3, 1, 0, 3, 1>)
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (Quaternion)arg1, (Quaternion)arg2) -> bool :
        
            C++ signature :
                bool __ne__(Eigen::Quaternion<double, 0>,Eigen::Quaternion<double, 0>)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (Quaternion)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > __repr__(Eigen::Quaternion<double, 0>)
        """
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (Quaternion)arg1, (int)arg2, (float)arg3) -> None :
        
            C++ signature :
                void __setitem__(Eigen::Quaternion<double, 0> {lvalue},int,double)
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (Quaternion)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > __str__(Eigen::Quaternion<double, 0>)
        """
    @staticmethod
    def _transformVector(*args, **kwargs):
        """
        
        _transformVector( (Quaternion)self, (numpy.ndarray)vector) -> numpy.ndarray :
            Rotation of a vector by a quaternion.
        
            C++ signature :
                Eigen::Matrix<double, 3, 1, 0, 3, 1> _transformVector(Eigen::Quaternion<double, 0> {lvalue},Eigen::Matrix<double, 3, 1, 0, 3, 1>)
        """
    @staticmethod
    def angularDistance(*args, **kwargs):
        """
        
        angularDistance( (Quaternion)arg1, (Quaternion)arg2) -> float :
            Returns the angle (in radian) between two rotations.
        
            C++ signature :
                double angularDistance(Eigen::Quaternion<double, 0> {lvalue},Eigen::QuaternionBase<Eigen::Quaternion<double, 0> >)
        """
    @staticmethod
    def assign(*args, **kwargs):
        """
        
        assign( (Quaternion)self, (Quaternion)quat) -> Quaternion :
            Set *this from an quaternion quat and returns a reference to *this.
        
            C++ signature :
                Eigen::Quaternion<double, 0> {lvalue} assign(Eigen::Quaternion<double, 0> {lvalue},Eigen::Quaternion<double, 0>)
        
        assign( (Quaternion)self, (AngleAxis)aa) -> Quaternion :
            Set *this from an angle-axis aa and returns a reference to *this.
        
            C++ signature :
                Eigen::Quaternion<double, 0> {lvalue} assign(Eigen::Quaternion<double, 0> {lvalue},Eigen::AngleAxis<double>)
        """
    @staticmethod
    def coeffs(*args, **kwargs):
        """
        
        coeffs( (Quaternion)self) -> object :
            Returns a vector of the coefficients (x,y,z,w)
        
            C++ signature :
                Eigen::Matrix<double, 4, 1, 0, 4, 1> coeffs(Eigen::Quaternion<double, 0> {lvalue})
        """
    @staticmethod
    def conjugate(*args, **kwargs):
        """
        
        conjugate( (Quaternion)self) -> Quaternion :
            Returns the conjugated quaternion.
            The conjugate of a quaternion represents the opposite rotation.
        
            C++ signature :
                Eigen::Quaternion<double, 0> conjugate(Eigen::Quaternion<double, 0> {lvalue})
        """
    @staticmethod
    def dot(*args, **kwargs):
        """
        
        dot( (Quaternion)self, (Quaternion)other) -> float :
            Returns the dot product of *this with an other Quaternion.
            Geometrically speaking, the dot product of two unit quaternions corresponds to the cosine of half the angle between the two rotations.
        
            C++ signature :
                double dot(Eigen::Quaternion<double, 0> {lvalue},Eigen::QuaternionBase<Eigen::Quaternion<double, 0> >)
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (Quaternion)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        
            C++ signature :
                long id(Eigen::Quaternion<double, 0>)
        """
    @staticmethod
    def inverse(*args, **kwargs):
        """
        
        inverse( (Quaternion)self) -> Quaternion :
            Returns the quaternion describing the inverse rotation.
        
            C++ signature :
                Eigen::Quaternion<double, 0> inverse(Eigen::Quaternion<double, 0> {lvalue})
        """
    @staticmethod
    def isApprox(*args, **kwargs):
        """
        
        isApprox( (Quaternion)self, (Quaternion)other [, (float)prec]) -> bool :
            Returns true if *this is approximately equal to other, within the precision determined by prec.
        
            C++ signature :
                bool isApprox(Eigen::Quaternion<double, 0>,Eigen::Quaternion<double, 0> [,double])
        """
    @staticmethod
    def matrix(*args, **kwargs):
        """
        
        matrix( (Quaternion)self) -> numpy.ndarray :
            Returns an equivalent 3x3 rotation matrix. Similar to toRotationMatrix.
        
            C++ signature :
                Eigen::Matrix<double, 3, 3, 0, 3, 3> matrix(Eigen::Quaternion<double, 0> {lvalue})
        """
    @staticmethod
    def norm(*args, **kwargs):
        """
        
        norm( (Quaternion)self) -> float :
            Returns the norm of the quaternion's coefficients.
        
            C++ signature :
                double norm(Eigen::Quaternion<double, 0> {lvalue})
        """
    @staticmethod
    def normalize(*args, **kwargs):
        """
        
        normalize( (Quaternion)self) -> Quaternion :
            Normalizes the quaternion *this.
        
            C++ signature :
                Eigen::Quaternion<double, 0> {lvalue} normalize(Eigen::Quaternion<double, 0> {lvalue})
        """
    @staticmethod
    def normalized(*args, **kwargs):
        """
        
        normalized( (Quaternion)self) -> Quaternion :
            Returns a normalized copy of *this.
        
            C++ signature :
                Eigen::Quaternion<double, 0>* normalized(Eigen::Quaternion<double, 0>)
        """
    @staticmethod
    def setFromTwoVectors(*args, **kwargs):
        """
        
        setFromTwoVectors( (Quaternion)self, (numpy.ndarray)a, (numpy.ndarray)b) -> Quaternion :
            Set *this to be the quaternion which transforms a into b through a rotation.
        
            C++ signature :
                Eigen::Quaternion<double, 0> {lvalue} setFromTwoVectors(Eigen::Quaternion<double, 0> {lvalue},Eigen::Matrix<double, 3, 1, 0, 3, 1>,Eigen::Matrix<double, 3, 1, 0, 3, 1>)
        """
    @staticmethod
    def setIdentity(*args, **kwargs):
        """
        
        setIdentity( (Quaternion)self) -> Quaternion :
            Set *this to the identity rotation.
        
            C++ signature :
                Eigen::Quaternion<double, 0> {lvalue} setIdentity(Eigen::Quaternion<double, 0> {lvalue})
        """
    @staticmethod
    def slerp(*args, **kwargs):
        """
        
        slerp( (Quaternion)self, (float)t, (Quaternion)other) -> Quaternion :
            Returns the spherical linear interpolation between the two quaternions *this and other at the parameter t in [0;1].
        
            C++ signature :
                Eigen::Quaternion<double, 0> slerp(Eigen::Quaternion<double, 0>,double,Eigen::Quaternion<double, 0>)
        """
    @staticmethod
    def squaredNorm(*args, **kwargs):
        """
        
        squaredNorm( (Quaternion)self) -> float :
            Returns the squared norm of the quaternion's coefficients.
        
            C++ signature :
                double squaredNorm(Eigen::Quaternion<double, 0> {lvalue})
        """
    @staticmethod
    def toRotationMatrix(*args, **kwargs):
        """
        
        toRotationMatrix( (Quaternion)arg1) -> numpy.ndarray :
            Returns an equivalent rotation matrix.
        
            C++ signature :
                Eigen::Matrix<double, 3, 3, 0, 3, 3> toRotationMatrix(Eigen::Quaternion<double, 0> {lvalue})
        """
    @staticmethod
    def vec(*args, **kwargs):
        """
        
        vec( (Quaternion)self) -> numpy.ndarray :
            Returns a vector expression of the imaginary part (x,y,z).
        
            C++ signature :
                Eigen::Matrix<double, 3, 1, 0, 3, 1> vec(Eigen::Quaternion<double, 0>)
        """
    @property
    def w(*args, **kwargs):
        """
        The w coefficient.
        """
    @w.setter
    def w(*args, **kwargs):
        ...
    @property
    def x(*args, **kwargs):
        """
        The x coefficient.
        """
    @x.setter
    def x(*args, **kwargs):
        ...
    @property
    def y(*args, **kwargs):
        """
        The y coefficient.
        """
    @y.setter
    def y(*args, **kwargs):
        ...
    @property
    def z(*args, **kwargs):
        """
        The z coefficient.
        """
    @z.setter
    def z(*args, **kwargs):
        ...
class RobotWrapper(Boost.Python.instance):
    """
    Robot Wrapper info.
    """
    @staticmethod
    def Jcom(*args, **kwargs):
        """
        
        Jcom( (RobotWrapper)arg1, (object)data) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, 3, -1, 0, 3, -1> Jcom(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>)
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)filename, (object)package_dir, (bool)verbose) -> None :
            Default constructor without RootJoint.
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > >,bool)
        
        __init__( (object)arg1, (str)filename, (object)package_dir, (object)roottype, (bool)verbose) -> None :
            Default constructor with RootJoint.
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > >,boost::variant<pinocchio::JointModelRevoluteTpl<double, 0, 0>, pinocchio::JointModelRevoluteTpl<double, 0, 1>, pinocchio::JointModelRevoluteTpl<double, 0, 2>, pinocchio::JointModelFreeFlyerTpl<double, 0>, pinocchio::JointModelPlanarTpl<double, 0>, pinocchio::JointModelRevoluteUnalignedTpl<double, 0>, pinocchio::JointModelSphericalTpl<double, 0>, pinocchio::JointModelSphericalZYXTpl<double, 0>, pinocchio::JointModelPrismaticTpl<double, 0, 0>, pinocchio::JointModelPrismaticTpl<double, 0, 1>, pinocchio::JointModelPrismaticTpl<double, 0, 2>, pinocchio::JointModelPrismaticUnalignedTpl<double, 0>, pinocchio::JointModelTranslationTpl<double, 0>, pinocchio::JointModelRevoluteUnboundedTpl<double, 0, 0>, pinocchio::JointModelRevoluteUnboundedTpl<double, 0, 1>, pinocchio::JointModelRevoluteUnboundedTpl<double, 0, 2>, pinocchio::JointModelRevoluteUnboundedUnalignedTpl<double, 0>, pinocchio::JointModelHelicalTpl<double, 0, 0>, pinocchio::JointModelHelicalTpl<double, 0, 1>, pinocchio::JointModelHelicalTpl<double, 0, 2>, pinocchio::JointModelHelicalUnalignedTpl<double, 0>, pinocchio::JointModelUniversalTpl<double, 0>, boost::recursive_wrapper<pinocchio::JointModelCompositeTpl<double, 0, pinocchio::JointCollectionDefaultTpl> >, boost::recursive_wrapper<pinocchio::JointModelMimicTpl<double, 0, pinocchio::JointCollectionDefaultTpl> > > {lvalue},bool)
        
        __init__( (object)arg1, (object)Pinocchio Model, (bool)verbose) -> None :
            Default constructor from pinocchio model without RootJoint.
        
            C++ signature :
                void __init__(_object*,pinocchio::ModelTpl<double, 0, pinocchio::JointCollectionDefaultTpl>,bool)
        
        __init__( (object)arg1, (object)Pinocchio Model, (RootJointType)rootJoint, (bool)verbose) -> None :
            Default constructor from pinocchio model with RootJoint.
        
            C++ signature :
                void __init__(_object*,pinocchio::ModelTpl<double, 0, pinocchio::JointCollectionDefaultTpl>,tsid::robots::RobotWrapper::e_RootJointType,bool)
        
        __init__( (object)arg1, (str)arg2, (object)arg3, (object)arg4, (bool)arg5) -> object :
        
            C++ signature :
                void* __init__(boost::python::api::object,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > >,boost::python::api::object {lvalue},bool)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def acceleration(*args, **kwargs):
        """
        
        acceleration( (RobotWrapper)arg1, (object)data, (int)index) -> object :
        
            C++ signature :
                pinocchio::MotionTpl<double, 0> acceleration(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>,unsigned long)
        """
    @staticmethod
    def angularMomentumTimeVariation(*args, **kwargs):
        """
        
        angularMomentumTimeVariation( (RobotWrapper)arg1, (object)data) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, 3, 1, 0, 3, 1> angularMomentumTimeVariation(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>)
        """
    @staticmethod
    def com(*args, **kwargs):
        """
        
        com( (RobotWrapper)arg1, (object)data) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, 3, 1, 0, 3, 1> com(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>)
        """
    @staticmethod
    def com_acc(*args, **kwargs):
        """
        
        com_acc( (RobotWrapper)arg1, (object)data) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, 3, 1, 0, 3, 1> com_acc(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>)
        """
    @staticmethod
    def com_vel(*args, **kwargs):
        """
        
        com_vel( (RobotWrapper)arg1, (object)data) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, 3, 1, 0, 3, 1> com_vel(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>)
        """
    @staticmethod
    def computeAllTerms(*args, **kwargs):
        """
        
        computeAllTerms( (RobotWrapper)arg1, (object)data, (numpy.ndarray)q, (numpy.ndarray)v) -> None :
            compute all dynamics
        
            C++ signature :
                void computeAllTerms(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def data(*args, **kwargs):
        """
        
        data( (RobotWrapper)arg1) -> object :
        
            C++ signature :
                pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> data(tsid::robots::RobotWrapper)
        """
    @staticmethod
    def frameAcceleration(*args, **kwargs):
        """
        
        frameAcceleration( (RobotWrapper)arg1, (object)data, (int)index) -> object :
        
            C++ signature :
                pinocchio::MotionTpl<double, 0> frameAcceleration(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>,unsigned long)
        """
    @staticmethod
    def frameAccelerationWorldOriented(*args, **kwargs):
        """
        
        frameAccelerationWorldOriented( (RobotWrapper)arg1, (object)data, (int)index) -> object :
        
            C++ signature :
                pinocchio::MotionTpl<double, 0> frameAccelerationWorldOriented(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>,unsigned long)
        """
    @staticmethod
    def frameClassicAcceleration(*args, **kwargs):
        """
        
        frameClassicAcceleration( (RobotWrapper)arg1, (object)data, (int)index) -> object :
        
            C++ signature :
                pinocchio::MotionTpl<double, 0> frameClassicAcceleration(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>,unsigned long)
        """
    @staticmethod
    def frameClassicAccelerationWorldOriented(*args, **kwargs):
        """
        
        frameClassicAccelerationWorldOriented( (RobotWrapper)arg1, (object)data, (int)index) -> object :
        
            C++ signature :
                pinocchio::MotionTpl<double, 0> frameClassicAccelerationWorldOriented(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>,unsigned long)
        """
    @staticmethod
    def framePosition(*args, **kwargs):
        """
        
        framePosition( (RobotWrapper)arg1, (object)data, (int)index) -> object :
        
            C++ signature :
                pinocchio::SE3Tpl<double, 0> framePosition(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>,unsigned long)
        """
    @staticmethod
    def frameVelocity(*args, **kwargs):
        """
        
        frameVelocity( (RobotWrapper)arg1, (object)data, (int)index) -> object :
        
            C++ signature :
                pinocchio::MotionTpl<double, 0> frameVelocity(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>,unsigned long)
        """
    @staticmethod
    def frameVelocityWorldOriented(*args, **kwargs):
        """
        
        frameVelocityWorldOriented( (RobotWrapper)arg1, (object)data, (int)index) -> object :
        
            C++ signature :
                pinocchio::MotionTpl<double, 0> frameVelocityWorldOriented(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>,unsigned long)
        """
    @staticmethod
    def mass(*args, **kwargs):
        """
        
        mass( (RobotWrapper)arg1, (object)data) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, -1, 0, -1, -1> mass(tsid::robots::RobotWrapper {lvalue},pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def model(*args, **kwargs):
        """
        
        model( (RobotWrapper)arg1) -> object :
        
            C++ signature :
                pinocchio::ModelTpl<double, 0, pinocchio::JointCollectionDefaultTpl> model(tsid::robots::RobotWrapper)
        """
    @staticmethod
    def nonLinearEffect(*args, **kwargs):
        """
        
        nonLinearEffect( (RobotWrapper)arg1, (object)data) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> nonLinearEffect(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>)
        """
    @staticmethod
    def position(*args, **kwargs):
        """
        
        position( (RobotWrapper)arg1, (object)data, (int)index) -> object :
        
            C++ signature :
                pinocchio::SE3Tpl<double, 0> position(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>,unsigned long)
        """
    @staticmethod
    def setGravity(*args, **kwargs):
        """
        
        setGravity( (RobotWrapper)arg1, (object)gravity) -> None :
        
            C++ signature :
                void setGravity(tsid::robots::RobotWrapper {lvalue},pinocchio::MotionTpl<double, 0>)
        """
    @staticmethod
    def set_gear_ratios(*args, **kwargs):
        """
        
        set_gear_ratios( (RobotWrapper)arg1, (numpy.ndarray)gear ratio vector) -> bool :
        
            C++ signature :
                bool set_gear_ratios(tsid::robots::RobotWrapper {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def set_rotor_inertias(*args, **kwargs):
        """
        
        set_rotor_inertias( (RobotWrapper)arg1, (numpy.ndarray)inertia vector) -> bool :
        
            C++ signature :
                bool set_rotor_inertias(tsid::robots::RobotWrapper {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def velocity(*args, **kwargs):
        """
        
        velocity( (RobotWrapper)arg1, (object)data, (int)index) -> object :
        
            C++ signature :
                pinocchio::MotionTpl<double, 0> velocity(tsid::robots::RobotWrapper,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl>,unsigned long)
        """
    @property
    def gear_ratios(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.RobotWrapper)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::robots::RobotWrapper)
        """
    @property
    def is_fixed_base(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.RobotWrapper)arg1) -> bool :
        
            C++ signature :
                bool None(tsid::robots::RobotWrapper {lvalue})
        """
    @property
    def na(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.RobotWrapper)arg1) -> int :
        
            C++ signature :
                int None(tsid::robots::RobotWrapper {lvalue})
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.RobotWrapper)arg1) -> int :
        
            C++ signature :
                int None(tsid::robots::RobotWrapper {lvalue})
        """
    @property
    def nq_actuated(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.RobotWrapper)arg1) -> int :
        
            C++ signature :
                int None(tsid::robots::RobotWrapper {lvalue})
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.RobotWrapper)arg1) -> int :
        
            C++ signature :
                int None(tsid::robots::RobotWrapper {lvalue})
        """
    @property
    def rotor_inertias(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.RobotWrapper)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::robots::RobotWrapper)
        """
class RootJointType(Boost.Python.enum):
    FIXED_BASE_SYSTEM: typing.ClassVar[RootJointType]  # value = tsid.tsid_pywrap.RootJointType.FIXED_BASE_SYSTEM
    FLOATING_BASE_SYSTEM: typing.ClassVar[RootJointType]  # value = tsid.tsid_pywrap.RootJointType.FLOATING_BASE_SYSTEM
    __slots__: typing.ClassVar[tuple] = tuple()
    names: typing.ClassVar[dict]  # value = {'FIXED_BASE_SYSTEM': tsid.tsid_pywrap.RootJointType.FIXED_BASE_SYSTEM, 'FLOATING_BASE_SYSTEM': tsid.tsid_pywrap.RootJointType.FLOATING_BASE_SYSTEM}
    values: typing.ClassVar[dict]  # value = {0: tsid.tsid_pywrap.RootJointType.FIXED_BASE_SYSTEM, 1: tsid.tsid_pywrap.RootJointType.FLOATING_BASE_SYSTEM}
class SolverHQuadProg(Boost.Python.instance):
    """
    Solver EiQuadProg info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name) -> None :
            Default Constructor with name
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def resize(*args, **kwargs):
        """
        
        resize( (SolverHQuadProg)arg1, (int)n, (int)neq, (int)nin) -> None :
        
            C++ signature :
                void resize(tsid::solvers::SolverHQuadProg {lvalue},unsigned int,unsigned int,unsigned int)
        """
    @staticmethod
    def retrieveQPData(*args, **kwargs):
        """
        
        retrieveQPData( (SolverHQuadProg)arg1, (object)arg2, (bool)HQPData) -> None :
        
            C++ signature :
                void retrieveQPData(tsid::solvers::SolverHQuadProg {lvalue},std::vector<std::vector<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> >, Eigen::aligned_allocator<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> > > >, Eigen::aligned_allocator<std::vector<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> >, Eigen::aligned_allocator<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> > > > > >,bool)
        
        retrieveQPData( (SolverHQuadProg)arg1, (HQPData)HQPData for Python) -> QPDataQuadProg :
        
            C++ signature :
                tsid::solvers::QPDataQuadProgTpl<double> retrieveQPData(tsid::solvers::SolverHQuadProg {lvalue},tsid::python::HQPDatas {lvalue})
        """
    @staticmethod
    def solve(*args, **kwargs):
        """
        
        solve( (SolverHQuadProg)arg1, (object)HQPData) -> HQPOutput :
        
            C++ signature :
                tsid::solvers::HQPOutput solve(tsid::solvers::SolverHQuadProg {lvalue},std::vector<std::vector<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> >, Eigen::aligned_allocator<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> > > >, Eigen::aligned_allocator<std::vector<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> >, Eigen::aligned_allocator<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> > > > > >)
        
        solve( (SolverHQuadProg)arg1, (HQPData)HQPData for Python) -> HQPOutput :
        
            C++ signature :
                tsid::solvers::HQPOutput solve(tsid::solvers::SolverHQuadProg {lvalue},tsid::python::HQPDatas {lvalue})
        """
    @property
    def ObjVal(*args, **kwargs):
        """
        return obj value
        """
    @property
    def qpData(*args, **kwargs):
        """
        return QP Data object
        """
class SolverHQuadProgFast(Boost.Python.instance):
    """
    Solver EiQuadProg info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name) -> None :
            Default Constructor with name
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def resize(*args, **kwargs):
        """
        
        resize( (SolverHQuadProgFast)arg1, (int)n, (int)neq, (int)nin) -> None :
        
            C++ signature :
                void resize(tsid::solvers::SolverHQuadProgFast {lvalue},unsigned int,unsigned int,unsigned int)
        """
    @staticmethod
    def retrieveQPData(*args, **kwargs):
        """
        
        retrieveQPData( (SolverHQuadProgFast)arg1, (object)arg2, (bool)HQPData) -> None :
        
            C++ signature :
                void retrieveQPData(tsid::solvers::SolverHQuadProgFast {lvalue},std::vector<std::vector<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> >, Eigen::aligned_allocator<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> > > >, Eigen::aligned_allocator<std::vector<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> >, Eigen::aligned_allocator<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> > > > > >,bool)
        
        retrieveQPData( (SolverHQuadProgFast)arg1, (HQPData)HQPData for Python) -> QPDataQuadProg :
        
            C++ signature :
                tsid::solvers::QPDataQuadProgTpl<double> retrieveQPData(tsid::solvers::SolverHQuadProgFast {lvalue},tsid::python::HQPDatas {lvalue})
        """
    @staticmethod
    def solve(*args, **kwargs):
        """
        
        solve( (SolverHQuadProgFast)arg1, (object)HQPData) -> HQPOutput :
        
            C++ signature :
                tsid::solvers::HQPOutput solve(tsid::solvers::SolverHQuadProgFast {lvalue},std::vector<std::vector<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> >, Eigen::aligned_allocator<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> > > >, Eigen::aligned_allocator<std::vector<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> >, Eigen::aligned_allocator<tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> > > > > >)
        
        solve( (SolverHQuadProgFast)arg1, (HQPData)HQPData for Python) -> HQPOutput :
        
            C++ signature :
                tsid::solvers::HQPOutput solve(tsid::solvers::SolverHQuadProgFast {lvalue},tsid::python::HQPDatas {lvalue})
        """
    @property
    def ObjVal(*args, **kwargs):
        """
        return obj value
        """
    @property
    def qpData(*args, **kwargs):
        """
        return QP Data object
        """
class TaskAMEquality(Boost.Python.instance):
    """
    TaskAMEqualityPythonVisitor info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue})
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def compute(*args, **kwargs):
        """
        
        compute( (TaskAMEquality)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality compute(tsid::tasks::TaskAMEquality {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def getConstraint(*args, **kwargs):
        """
        
        getConstraint( (TaskAMEquality)arg1) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality getConstraint(tsid::tasks::TaskAMEquality)
        """
    @staticmethod
    def getdMomentum(*args, **kwargs):
        """
        
        getdMomentum( (TaskAMEquality)arg1, (numpy.ndarray)dv) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, 3, 1, 0, 3, 1> getdMomentum(tsid::tasks::TaskAMEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setKd(*args, **kwargs):
        """
        
        setKd( (TaskAMEquality)arg1, (numpy.ndarray)Kd) -> None :
        
            C++ signature :
                void setKd(tsid::tasks::TaskAMEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setKp(*args, **kwargs):
        """
        
        setKp( (TaskAMEquality)arg1, (numpy.ndarray)Kp) -> None :
        
            C++ signature :
                void setKp(tsid::tasks::TaskAMEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setReference(*args, **kwargs):
        """
        
        setReference( (TaskAMEquality)arg1, (TrajectorySample)ref) -> None :
        
            C++ signature :
                void setReference(tsid::tasks::TaskAMEquality {lvalue},tsid::trajectories::TrajectorySample)
        """
    @property
    def Kd(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskAMEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, 3, 1, 0, 3, 1> None(tsid::tasks::TaskAMEquality {lvalue})
        """
    @property
    def Kp(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskAMEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, 3, 1, 0, 3, 1> None(tsid::tasks::TaskAMEquality {lvalue})
        """
    @property
    def dim(*args, **kwargs):
        """
        return dimension size
        """
    @property
    def dmomentum_ref(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskAMEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskAMEquality)
        """
    @property
    def getDesiredMomentumDerivative(*args, **kwargs):
        """
        Return dL_desired
        """
    @property
    def momentum(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskAMEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, 3, 1, 0, 3, 1> None(tsid::tasks::TaskAMEquality)
        """
    @property
    def momentum_error(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskAMEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, 3, 1, 0, 3, 1> None(tsid::tasks::TaskAMEquality)
        """
    @property
    def momentum_ref(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskAMEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskAMEquality)
        """
    @property
    def name(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskAMEquality)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > None(tsid::tasks::TaskAMEquality {lvalue})
        """
class TaskActuationBounds(Boost.Python.instance):
    """
    Task info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue})
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def compute(*args, **kwargs):
        """
        
        compute( (TaskActuationBounds)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintInequality :
        
            C++ signature :
                tsid::math::ConstraintInequality compute(tsid::tasks::TaskActuationBounds {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def getConstraint(*args, **kwargs):
        """
        
        getConstraint( (TaskActuationBounds)arg1) -> ConstraintInequality :
        
            C++ signature :
                tsid::math::ConstraintInequality getConstraint(tsid::tasks::TaskActuationBounds)
        """
    @staticmethod
    def setBounds(*args, **kwargs):
        """
        
        setBounds( (TaskActuationBounds)arg1, (numpy.ndarray)lower, (numpy.ndarray)upper) -> None :
        
            C++ signature :
                void setBounds(tsid::tasks::TaskActuationBounds {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setMask(*args, **kwargs):
        """
        
        setMask( (TaskActuationBounds)arg1, (numpy.ndarray)mask) -> None :
        
            C++ signature :
                void setMask(tsid::tasks::TaskActuationBounds {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @property
    def dim(*args, **kwargs):
        """
        return dimension size
        """
    @property
    def getLowerBounds(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskActuationBounds)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskActuationBounds)
        """
    @property
    def getUpperBounds(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskActuationBounds)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskActuationBounds)
        """
    @property
    def mask(*args, **kwargs):
        """
        Return mask
        """
    @property
    def name(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskActuationBounds)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > None(tsid::tasks::TaskActuationBounds {lvalue})
        """
class TaskActuationEquality(Boost.Python.instance):
    """
    TaskActuationEqualityPythonVisitor info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue})
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def compute(*args, **kwargs):
        """
        
        compute( (TaskActuationEquality)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality compute(tsid::tasks::TaskActuationEquality {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def getConstraint(*args, **kwargs):
        """
        
        getConstraint( (TaskActuationEquality)arg1) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality getConstraint(tsid::tasks::TaskActuationEquality)
        """
    @staticmethod
    def getReference(*args, **kwargs):
        """
        
        getReference( (TaskActuationEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> getReference(tsid::tasks::TaskActuationEquality)
        """
    @staticmethod
    def getWeightVector(*args, **kwargs):
        """
        
        getWeightVector( (TaskActuationEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> getWeightVector(tsid::tasks::TaskActuationEquality)
        """
    @staticmethod
    def setMask(*args, **kwargs):
        """
        
        setMask( (TaskActuationEquality)arg1, (numpy.ndarray)mask) -> None :
        
            C++ signature :
                void setMask(tsid::tasks::TaskActuationEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setReference(*args, **kwargs):
        """
        
        setReference( (TaskActuationEquality)arg1, (numpy.ndarray)ref) -> None :
        
            C++ signature :
                void setReference(tsid::tasks::TaskActuationEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setWeightVector(*args, **kwargs):
        """
        
        setWeightVector( (TaskActuationEquality)arg1, (numpy.ndarray)weights) -> None :
        
            C++ signature :
                void setWeightVector(tsid::tasks::TaskActuationEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @property
    def dim(*args, **kwargs):
        """
        return dimension size
        """
    @property
    def mask(*args, **kwargs):
        """
        Return mask
        """
class TaskComEquality(Boost.Python.instance):
    """
    TaskCOMEqualityPythonVisitor info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue})
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def compute(*args, **kwargs):
        """
        
        compute( (TaskComEquality)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality compute(tsid::tasks::TaskComEquality {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def getAcceleration(*args, **kwargs):
        """
        
        getAcceleration( (TaskComEquality)arg1, (numpy.ndarray)dv) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> getAcceleration(tsid::tasks::TaskComEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def getConstraint(*args, **kwargs):
        """
        
        getConstraint( (TaskComEquality)arg1) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality getConstraint(tsid::tasks::TaskComEquality)
        """
    @staticmethod
    def setKd(*args, **kwargs):
        """
        
        setKd( (TaskComEquality)arg1, (numpy.ndarray)Kd) -> None :
        
            C++ signature :
                void setKd(tsid::tasks::TaskComEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setKp(*args, **kwargs):
        """
        
        setKp( (TaskComEquality)arg1, (numpy.ndarray)Kp) -> None :
        
            C++ signature :
                void setKp(tsid::tasks::TaskComEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setMask(*args, **kwargs):
        """
        
        setMask( (TaskComEquality)arg1, (numpy.ndarray)mask) -> None :
        
            C++ signature :
                void setMask(tsid::tasks::TaskComEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setReference(*args, **kwargs):
        """
        
        setReference( (TaskComEquality)arg1, (TrajectorySample)ref) -> None :
        
            C++ signature :
                void setReference(tsid::tasks::TaskComEquality {lvalue},tsid::trajectories::TrajectorySample)
        """
    @property
    def Kd(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskComEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, 3, 1, 0, 3, 1> None(tsid::tasks::TaskComEquality {lvalue})
        """
    @property
    def Kp(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskComEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, 3, 1, 0, 3, 1> None(tsid::tasks::TaskComEquality {lvalue})
        """
    @property
    def dim(*args, **kwargs):
        """
        return dimension size
        """
    @property
    def getDesiredAcceleration(*args, **kwargs):
        """
        Return Acc_desired
        """
    @property
    def mask(*args, **kwargs):
        """
        Return mask
        """
    @property
    def name(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskComEquality)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > None(tsid::tasks::TaskComEquality {lvalue})
        """
    @property
    def position(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskComEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskComEquality)
        """
    @property
    def position_error(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskComEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskComEquality)
        """
    @property
    def position_ref(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskComEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskComEquality)
        """
    @property
    def velocity(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskComEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskComEquality)
        """
    @property
    def velocity_error(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskComEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskComEquality)
        """
    @property
    def velocity_ref(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskComEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskComEquality)
        """
class TaskCopEquality(Boost.Python.instance):
    """
    TaskCOPEqualityPythonVisitor info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue})
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def compute(*args, **kwargs):
        """
        
        compute( (TaskCopEquality)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality compute(tsid::tasks::TaskCopEquality {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def getConstraint(*args, **kwargs):
        """
        
        getConstraint( (TaskCopEquality)arg1) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality getConstraint(tsid::tasks::TaskCopEquality)
        """
    @staticmethod
    def setContactNormal(*args, **kwargs):
        """
        
        setContactNormal( (TaskCopEquality)arg1, (numpy.ndarray)normal) -> None :
        
            C++ signature :
                void setContactNormal(tsid::tasks::TaskCopEquality {lvalue},Eigen::Matrix<double, 3, 1, 0, 3, 1>)
        """
    @staticmethod
    def setReference(*args, **kwargs):
        """
        
        setReference( (TaskCopEquality)arg1, (numpy.ndarray)ref) -> None :
        
            C++ signature :
                void setReference(tsid::tasks::TaskCopEquality {lvalue},Eigen::Matrix<double, 3, 1, 0, 3, 1>)
        """
    @property
    def dim(*args, **kwargs):
        """
        return dimension size
        """
    @property
    def name(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskCopEquality)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > None(tsid::tasks::TaskCopEquality {lvalue})
        """
class TaskJointBounds(Boost.Python.instance):
    """
    Task info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot, (float)Time step) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue},double)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def compute(*args, **kwargs):
        """
        
        compute( (TaskJointBounds)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintBound :
        
            C++ signature :
                tsid::math::ConstraintBound compute(tsid::tasks::TaskJointBounds {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def getConstraint(*args, **kwargs):
        """
        
        getConstraint( (TaskJointBounds)arg1) -> ConstraintBound :
        
            C++ signature :
                tsid::math::ConstraintBound getConstraint(tsid::tasks::TaskJointBounds)
        """
    @staticmethod
    def setAccelerationBounds(*args, **kwargs):
        """
        
        setAccelerationBounds( (TaskJointBounds)arg1, (numpy.ndarray)lower, (numpy.ndarray)upper) -> None :
        
            C++ signature :
                void setAccelerationBounds(tsid::tasks::TaskJointBounds {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setTimeStep(*args, **kwargs):
        """
        
        setTimeStep( (TaskJointBounds)arg1, (float)dt) -> None :
        
            C++ signature :
                void setTimeStep(tsid::tasks::TaskJointBounds {lvalue},double)
        """
    @staticmethod
    def setVelocityBounds(*args, **kwargs):
        """
        
        setVelocityBounds( (TaskJointBounds)arg1, (numpy.ndarray)lower, (numpy.ndarray)upper) -> None :
        
            C++ signature :
                void setVelocityBounds(tsid::tasks::TaskJointBounds {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @property
    def dim(*args, **kwargs):
        """
        return dimension size
        """
    @property
    def getAccelerationLowerBounds(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointBounds)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointBounds)
        """
    @property
    def getAccelerationUpperBounds(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointBounds)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointBounds)
        """
    @property
    def getVelocityLowerBounds(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointBounds)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointBounds)
        """
    @property
    def getVelocityUpperBounds(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointBounds)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointBounds)
        """
    @property
    def name(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointBounds)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > None(tsid::tasks::TaskJointBounds {lvalue})
        """
class TaskJointPosVelAccBounds(Boost.Python.instance):
    """
    Task info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot, (float)Time step [, (bool)verbose]) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue},double [,bool])
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def compute(*args, **kwargs):
        """
        
        compute( (TaskJointPosVelAccBounds)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintBound :
        
            C++ signature :
                tsid::math::ConstraintBound compute(tsid::tasks::TaskJointPosVelAccBounds {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def computeAccLimits(*args, **kwargs):
        """
        
        computeAccLimits( (TaskJointPosVelAccBounds)arg1, (numpy.ndarray)q, (numpy.ndarray)dq [, (bool)verbose=True]) -> None :
        
            C++ signature :
                void computeAccLimits(tsid::tasks::TaskJointPosVelAccBounds {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >,Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > [,bool=True])
        """
    @staticmethod
    def computeAccLimitsFromPosLimits(*args, **kwargs):
        """
        
        computeAccLimitsFromPosLimits( (TaskJointPosVelAccBounds)arg1, (numpy.ndarray)q, (numpy.ndarray)dq [, (bool)verbose=True]) -> None :
        
            C++ signature :
                void computeAccLimitsFromPosLimits(tsid::tasks::TaskJointPosVelAccBounds {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >,Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > [,bool=True])
        """
    @staticmethod
    def computeAccLimitsFromViability(*args, **kwargs):
        """
        
        computeAccLimitsFromViability( (TaskJointPosVelAccBounds)arg1, (numpy.ndarray)q, (numpy.ndarray)dq [, (bool)verbose=True]) -> None :
        
            C++ signature :
                void computeAccLimitsFromViability(tsid::tasks::TaskJointPosVelAccBounds {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >,Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > [,bool=True])
        """
    @staticmethod
    def getConstraint(*args, **kwargs):
        """
        
        getConstraint( (TaskJointPosVelAccBounds)arg1) -> ConstraintBound :
        
            C++ signature :
                tsid::math::ConstraintBound getConstraint(tsid::tasks::TaskJointPosVelAccBounds)
        """
    @staticmethod
    def isStateViable(*args, **kwargs):
        """
        
        isStateViable( (TaskJointPosVelAccBounds)arg1, (numpy.ndarray)q, (numpy.ndarray)dq [, (bool)verbose=True]) -> None :
        
            C++ signature :
                void isStateViable(tsid::tasks::TaskJointPosVelAccBounds {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >,Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > [,bool=True])
        """
    @staticmethod
    def setAccelerationBounds(*args, **kwargs):
        """
        
        setAccelerationBounds( (TaskJointPosVelAccBounds)arg1, (numpy.ndarray)upper) -> None :
        
            C++ signature :
                void setAccelerationBounds(tsid::tasks::TaskJointPosVelAccBounds {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setImposeBounds(*args, **kwargs):
        """
        
        setImposeBounds( (TaskJointPosVelAccBounds)arg1, (bool)impose_position_bounds, (bool)impose_velocity_bounds, (bool)impose_viability_bounds, (bool)impose_acceleration_bounds) -> None :
        
            C++ signature :
                void setImposeBounds(tsid::tasks::TaskJointPosVelAccBounds {lvalue},bool,bool,bool,bool)
        """
    @staticmethod
    def setMask(*args, **kwargs):
        """
        
        setMask( (TaskJointPosVelAccBounds)arg1, (numpy.ndarray)mask) -> None :
        
            C++ signature :
                void setMask(tsid::tasks::TaskJointPosVelAccBounds {lvalue},Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> >)
        """
    @staticmethod
    def setPositionBounds(*args, **kwargs):
        """
        
        setPositionBounds( (TaskJointPosVelAccBounds)arg1, (numpy.ndarray)lower, (numpy.ndarray)upper) -> None :
        
            C++ signature :
                void setPositionBounds(tsid::tasks::TaskJointPosVelAccBounds {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setTimeStep(*args, **kwargs):
        """
        
        setTimeStep( (TaskJointPosVelAccBounds)arg1, (float)dt) -> None :
        
            C++ signature :
                void setTimeStep(tsid::tasks::TaskJointPosVelAccBounds {lvalue},double)
        """
    @staticmethod
    def setVelocityBounds(*args, **kwargs):
        """
        
        setVelocityBounds( (TaskJointPosVelAccBounds)arg1, (numpy.ndarray)upper) -> None :
        
            C++ signature :
                void setVelocityBounds(tsid::tasks::TaskJointPosVelAccBounds {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setVerbose(*args, **kwargs):
        """
        
        setVerbose( (TaskJointPosVelAccBounds)arg1, (bool)verbose) -> None :
        
            C++ signature :
                void setVerbose(tsid::tasks::TaskJointPosVelAccBounds {lvalue},bool)
        """
    @property
    def dim(*args, **kwargs):
        """
        return dimension size
        """
    @property
    def getAccelerationBounds(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosVelAccBounds)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointPosVelAccBounds)
        """
    @property
    def getPositionLowerBounds(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosVelAccBounds)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointPosVelAccBounds)
        """
    @property
    def getPositionUpperBounds(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosVelAccBounds)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointPosVelAccBounds)
        """
    @property
    def getVelocityBounds(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosVelAccBounds)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointPosVelAccBounds)
        """
    @property
    def name(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosVelAccBounds)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > None(tsid::tasks::TaskJointPosVelAccBounds {lvalue})
        """
class TaskJointPosture(Boost.Python.instance):
    """
    TaskJoint info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue})
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def compute(*args, **kwargs):
        """
        
        compute( (TaskJointPosture)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality compute(tsid::tasks::TaskJointPosture {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def getAcceleration(*args, **kwargs):
        """
        
        getAcceleration( (TaskJointPosture)arg1, (numpy.ndarray)dv) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> getAcceleration(tsid::tasks::TaskJointPosture {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def getConstraint(*args, **kwargs):
        """
        
        getConstraint( (TaskJointPosture)arg1) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality getConstraint(tsid::tasks::TaskJointPosture)
        """
    @staticmethod
    def setKd(*args, **kwargs):
        """
        
        setKd( (TaskJointPosture)arg1, (numpy.ndarray)Kd) -> None :
        
            C++ signature :
                void setKd(tsid::tasks::TaskJointPosture {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setKp(*args, **kwargs):
        """
        
        setKp( (TaskJointPosture)arg1, (numpy.ndarray)Kp) -> None :
        
            C++ signature :
                void setKp(tsid::tasks::TaskJointPosture {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setMask(*args, **kwargs):
        """
        
        setMask( (TaskJointPosture)arg1, (numpy.ndarray)mask) -> None :
        
            C++ signature :
                void setMask(tsid::tasks::TaskJointPosture {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setReference(*args, **kwargs):
        """
        
        setReference( (TaskJointPosture)arg1, (TrajectorySample)ref) -> None :
        
            C++ signature :
                void setReference(tsid::tasks::TaskJointPosture {lvalue},tsid::trajectories::TrajectorySample)
        """
    @property
    def Kd(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosture)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointPosture {lvalue})
        """
    @property
    def Kp(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosture)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointPosture {lvalue})
        """
    @property
    def dim(*args, **kwargs):
        """
        return dimension size
        """
    @property
    def getDesiredAcceleration(*args, **kwargs):
        """
        Return Acc_desired
        """
    @property
    def mask(*args, **kwargs):
        """
        Return mask
        """
    @property
    def name(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosture)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > None(tsid::tasks::TaskJointPosture {lvalue})
        """
    @property
    def position(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosture)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointPosture)
        """
    @property
    def position_error(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosture)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointPosture)
        """
    @property
    def position_ref(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosture)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointPosture)
        """
    @property
    def velocity(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosture)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointPosture)
        """
    @property
    def velocity_error(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosture)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointPosture)
        """
    @property
    def velocity_ref(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskJointPosture)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskJointPosture)
        """
class TaskSE3Equality(Boost.Python.instance):
    """
    TaskSE3 info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot, (str)framename) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def compute(*args, **kwargs):
        """
        
        compute( (TaskSE3Equality)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality compute(tsid::tasks::TaskSE3Equality {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def getAcceleration(*args, **kwargs):
        """
        
        getAcceleration( (TaskSE3Equality)arg1, (numpy.ndarray)dv) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> getAcceleration(tsid::tasks::TaskSE3Equality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def getConstraint(*args, **kwargs):
        """
        
        getConstraint( (TaskSE3Equality)arg1) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality getConstraint(tsid::tasks::TaskSE3Equality)
        """
    @staticmethod
    def setKd(*args, **kwargs):
        """
        
        setKd( (TaskSE3Equality)arg1, (numpy.ndarray)Kd) -> None :
        
            C++ signature :
                void setKd(tsid::tasks::TaskSE3Equality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setKp(*args, **kwargs):
        """
        
        setKp( (TaskSE3Equality)arg1, (numpy.ndarray)Kp) -> None :
        
            C++ signature :
                void setKp(tsid::tasks::TaskSE3Equality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setMask(*args, **kwargs):
        """
        
        setMask( (TaskSE3Equality)arg1, (numpy.ndarray)mask) -> None :
        
            C++ signature :
                void setMask(tsid::tasks::TaskSE3Equality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setReference(*args, **kwargs):
        """
        
        setReference( (TaskSE3Equality)arg1, (TrajectorySample)ref) -> None :
        
            C++ signature :
                void setReference(tsid::tasks::TaskSE3Equality {lvalue},tsid::trajectories::TrajectorySample {lvalue})
        """
    @staticmethod
    def useLocalFrame(*args, **kwargs):
        """
        
        useLocalFrame( (TaskSE3Equality)arg1, (bool)local_frame) -> None :
        
            C++ signature :
                void useLocalFrame(tsid::tasks::TaskSE3Equality {lvalue},bool)
        """
    @property
    def Kd(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskSE3Equality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskSE3Equality {lvalue})
        """
    @property
    def Kp(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskSE3Equality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskSE3Equality {lvalue})
        """
    @property
    def dim(*args, **kwargs):
        """
        return dimension size
        """
    @property
    def frame_id(*args, **kwargs):
        """
        frame id return
        """
    @property
    def getDesiredAcceleration(*args, **kwargs):
        """
        Return Acc_desired
        """
    @property
    def mask(*args, **kwargs):
        """
        Return mask
        """
    @property
    def name(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskSE3Equality)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > None(tsid::tasks::TaskSE3Equality {lvalue})
        """
    @property
    def position(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskSE3Equality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskSE3Equality)
        """
    @property
    def position_error(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskSE3Equality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskSE3Equality)
        """
    @property
    def position_ref(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskSE3Equality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskSE3Equality)
        """
    @property
    def velocity(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskSE3Equality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskSE3Equality)
        """
    @property
    def velocity_error(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskSE3Equality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskSE3Equality)
        """
    @property
    def velocity_ref(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskSE3Equality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskSE3Equality)
        """
class TaskTwoFramesEquality(Boost.Python.instance):
    """
    TaskFrames info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name, (RobotWrapper)robot, (str)framename1, (str)framename2) -> None :
            Default Constructor
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,tsid::robots::RobotWrapper {lvalue},std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def compute(*args, **kwargs):
        """
        
        compute( (TaskTwoFramesEquality)arg1, (float)t, (numpy.ndarray)q, (numpy.ndarray)v, (object)data) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality compute(tsid::tasks::TaskTwoFramesEquality {lvalue},double,Eigen::Matrix<double, -1, 1, 0, -1, 1>,Eigen::Matrix<double, -1, 1, 0, -1, 1>,pinocchio::DataTpl<double, 0, pinocchio::JointCollectionDefaultTpl> {lvalue})
        """
    @staticmethod
    def getAcceleration(*args, **kwargs):
        """
        
        getAcceleration( (TaskTwoFramesEquality)arg1, (numpy.ndarray)dv) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> getAcceleration(tsid::tasks::TaskTwoFramesEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def getConstraint(*args, **kwargs):
        """
        
        getConstraint( (TaskTwoFramesEquality)arg1) -> ConstraintEquality :
        
            C++ signature :
                tsid::math::ConstraintEquality getConstraint(tsid::tasks::TaskTwoFramesEquality)
        """
    @staticmethod
    def setKd(*args, **kwargs):
        """
        
        setKd( (TaskTwoFramesEquality)arg1, (numpy.ndarray)Kd) -> None :
        
            C++ signature :
                void setKd(tsid::tasks::TaskTwoFramesEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setKp(*args, **kwargs):
        """
        
        setKp( (TaskTwoFramesEquality)arg1, (numpy.ndarray)Kp) -> None :
        
            C++ signature :
                void setKp(tsid::tasks::TaskTwoFramesEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def setMask(*args, **kwargs):
        """
        
        setMask( (TaskTwoFramesEquality)arg1, (numpy.ndarray)mask) -> None :
        
            C++ signature :
                void setMask(tsid::tasks::TaskTwoFramesEquality {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @property
    def Kd(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskTwoFramesEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskTwoFramesEquality {lvalue})
        """
    @property
    def Kp(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskTwoFramesEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskTwoFramesEquality {lvalue})
        """
    @property
    def dim(*args, **kwargs):
        """
        return dimension size
        """
    @property
    def frame_id1(*args, **kwargs):
        """
        frame id 1 return
        """
    @property
    def frame_id2(*args, **kwargs):
        """
        frame id 2 return
        """
    @property
    def getDesiredAcceleration(*args, **kwargs):
        """
        Return Acc_desired
        """
    @property
    def mask(*args, **kwargs):
        """
        Return mask
        """
    @property
    def name(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskTwoFramesEquality)arg1) -> str :
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > None(tsid::tasks::TaskTwoFramesEquality {lvalue})
        """
    @property
    def position_error(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskTwoFramesEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskTwoFramesEquality)
        """
    @property
    def velocity_error(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TaskTwoFramesEquality)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> None(tsid::tasks::TaskTwoFramesEquality)
        """
class TrajectoryEuclidianConstant(Boost.Python.instance):
    """
    Trajectory Euclidian Constant info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name) -> None :
            Default Constructor with name
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
        
        __init__( (object)arg1, (str)name, (numpy.ndarray)reference) -> None :
            Default Constructor with name and ref_vec
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def computeNext(*args, **kwargs):
        """
        
        computeNext( (TrajectoryEuclidianConstant)arg1) -> TrajectorySample :
        
            C++ signature :
                tsid::trajectories::TrajectorySample computeNext(tsid::trajectories::TrajectoryEuclidianConstant {lvalue})
        """
    @staticmethod
    def getLastSample(*args, **kwargs):
        """
        
        getLastSample( (TrajectoryEuclidianConstant)arg1, (TrajectorySample)sample) -> None :
        
            C++ signature :
                void getLastSample(tsid::trajectories::TrajectoryEuclidianConstant,tsid::trajectories::TrajectorySample {lvalue})
        """
    @staticmethod
    def getSample(*args, **kwargs):
        """
        
        getSample( (TrajectoryEuclidianConstant)arg1, (float)time) -> TrajectorySample :
        
            C++ signature :
                tsid::trajectories::TrajectorySample getSample(tsid::trajectories::TrajectoryEuclidianConstant {lvalue},double)
        """
    @staticmethod
    def has_trajectory_ended(*args, **kwargs):
        """
        
        has_trajectory_ended( (TrajectoryEuclidianConstant)arg1) -> bool :
        
            C++ signature :
                bool has_trajectory_ended(tsid::trajectories::TrajectoryEuclidianConstant)
        """
    @staticmethod
    def setReference(*args, **kwargs):
        """
        
        setReference( (TrajectoryEuclidianConstant)arg1, (numpy.ndarray)ref_vec) -> None :
        
            C++ signature :
                void setReference(tsid::trajectories::TrajectoryEuclidianConstant {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @property
    def size(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TrajectoryEuclidianConstant)arg1) -> int :
        
            C++ signature :
                unsigned int None(tsid::trajectories::TrajectoryEuclidianConstant {lvalue})
        """
class TrajectorySE3Constant(Boost.Python.instance):
    """
    Trajectory SE3 Constant info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)name) -> None :
            Default Constructor with name
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
        
        __init__( (object)arg1, (str)name, (object)reference) -> None :
            Default Constructor with name and ref_vec
        
            C++ signature :
                void __init__(_object*,std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >,pinocchio::SE3Tpl<double, 0>)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def computeNext(*args, **kwargs):
        """
        
        computeNext( (TrajectorySE3Constant)arg1) -> TrajectorySample :
        
            C++ signature :
                tsid::trajectories::TrajectorySample computeNext(tsid::trajectories::TrajectorySE3Constant {lvalue})
        """
    @staticmethod
    def getLastSample(*args, **kwargs):
        """
        
        getLastSample( (TrajectorySE3Constant)arg1, (TrajectorySample)sample) -> None :
        
            C++ signature :
                void getLastSample(tsid::trajectories::TrajectorySE3Constant,tsid::trajectories::TrajectorySample {lvalue})
        """
    @staticmethod
    def getSample(*args, **kwargs):
        """
        
        getSample( (TrajectorySE3Constant)arg1, (float)time) -> TrajectorySample :
        
            C++ signature :
                tsid::trajectories::TrajectorySample getSample(tsid::trajectories::TrajectorySE3Constant {lvalue},double)
        """
    @staticmethod
    def has_trajectory_ended(*args, **kwargs):
        """
        
        has_trajectory_ended( (TrajectorySE3Constant)arg1) -> bool :
        
            C++ signature :
                bool has_trajectory_ended(tsid::trajectories::TrajectorySE3Constant)
        """
    @staticmethod
    def setReference(*args, **kwargs):
        """
        
        setReference( (TrajectorySE3Constant)arg1, (object)M_ref) -> None :
        
            C++ signature :
                void setReference(tsid::trajectories::TrajectorySE3Constant {lvalue},pinocchio::SE3Tpl<double, 0>)
        """
    @property
    def size(*args, **kwargs):
        """
        
        None( (tsid.tsid_pywrap.TrajectorySE3Constant)arg1) -> int :
        
            C++ signature :
                unsigned int None(tsid::trajectories::TrajectorySE3Constant {lvalue})
        """
class TrajectorySample(Boost.Python.instance):
    """
    Trajectory Sample info.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (int)size) -> None :
            Default Constructor with size
        
            C++ signature :
                void __init__(_object*,unsigned int)
        
        __init__( (object)arg1, (int)value_size, (int)derivative_size) -> None :
            Default Constructor with value and derivative size
        
            C++ signature :
                void __init__(_object*,unsigned int,unsigned int)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def acc(*args, **kwargs):
        """
        
        acc( (TrajectorySample)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> acc(tsid::trajectories::TrajectorySample)
        
        acc( (TrajectorySample)arg1, (numpy.ndarray)arg2) -> None :
        
            C++ signature :
                void acc(tsid::trajectories::TrajectorySample {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def derivative(*args, **kwargs):
        """
        
        derivative( (TrajectorySample)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> derivative(tsid::trajectories::TrajectorySample)
        
        derivative( (TrajectorySample)arg1, (numpy.ndarray)arg2) -> None :
        
            C++ signature :
                void derivative(tsid::trajectories::TrajectorySample {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def pos(*args, **kwargs):
        """
        
        pos( (TrajectorySample)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> pos(tsid::trajectories::TrajectorySample)
        
        pos( (TrajectorySample)arg1, (numpy.ndarray)arg2) -> None :
        
            C++ signature :
                void pos(tsid::trajectories::TrajectorySample {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        
        pos( (TrajectorySample)arg1, (object)arg2) -> None :
        
            C++ signature :
                void pos(tsid::trajectories::TrajectorySample {lvalue},pinocchio::SE3Tpl<double, 0>)
        """
    @staticmethod
    def resize(*args, **kwargs):
        """
        
        resize( (TrajectorySample)arg1, (int)size) -> None :
        
            C++ signature :
                void resize(tsid::trajectories::TrajectorySample {lvalue},unsigned int)
        
        resize( (TrajectorySample)arg1, (int)value_size, (int)derivative_size) -> None :
        
            C++ signature :
                void resize(tsid::trajectories::TrajectorySample {lvalue},unsigned int,unsigned int)
        """
    @staticmethod
    def second_derivative(*args, **kwargs):
        """
        
        second_derivative( (TrajectorySample)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> second_derivative(tsid::trajectories::TrajectorySample)
        
        second_derivative( (TrajectorySample)arg1, (numpy.ndarray)arg2) -> None :
        
            C++ signature :
                void second_derivative(tsid::trajectories::TrajectorySample {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
    @staticmethod
    def value(*args, **kwargs):
        """
        
        value( (TrajectorySample)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> value(tsid::trajectories::TrajectorySample)
        
        value( (TrajectorySample)arg1, (numpy.ndarray)arg2) -> None :
        
            C++ signature :
                void value(tsid::trajectories::TrajectorySample {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        
        value( (TrajectorySample)arg1, (object)arg2) -> None :
        
            C++ signature :
                void value(tsid::trajectories::TrajectorySample {lvalue},pinocchio::SE3Tpl<double, 0>)
        """
    @staticmethod
    def vel(*args, **kwargs):
        """
        
        vel( (TrajectorySample)arg1) -> numpy.ndarray :
        
            C++ signature :
                Eigen::Matrix<double, -1, 1, 0, -1, 1> vel(tsid::trajectories::TrajectorySample)
        
        vel( (TrajectorySample)arg1, (numpy.ndarray)arg2) -> None :
        
            C++ signature :
                void vel(tsid::trajectories::TrajectorySample {lvalue},Eigen::Matrix<double, -1, 1, 0, -1, 1>)
        """
class boost_type_index(Boost.Python.instance):
    """
    The class type_index holds implementation-specific information about a type, including the name of the type and means to compare two types for equality or collating order.
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (boost_type_index)arg1, (boost_type_index)arg2) -> object :
        
            C++ signature :
                _object* __eq__(boost::typeindex::stl_type_index {lvalue},boost::typeindex::stl_type_index)
        """
    @staticmethod
    def __ge__(*args, **kwargs):
        """
        
        __ge__( (boost_type_index)arg1, (boost_type_index)arg2) -> object :
        
            C++ signature :
                _object* __ge__(boost::typeindex::stl_type_index {lvalue},boost::typeindex::stl_type_index)
        """
    @staticmethod
    def __gt__(*args, **kwargs):
        """
        
        __gt__( (boost_type_index)arg1, (boost_type_index)arg2) -> object :
        
            C++ signature :
                _object* __gt__(boost::typeindex::stl_type_index {lvalue},boost::typeindex::stl_type_index)
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        Raises an exception
        This class cannot be instantiated from Python
        """
    @staticmethod
    def __le__(*args, **kwargs):
        """
        
        __le__( (boost_type_index)arg1, (boost_type_index)arg2) -> object :
        
            C++ signature :
                _object* __le__(boost::typeindex::stl_type_index {lvalue},boost::typeindex::stl_type_index)
        """
    @staticmethod
    def __lt__(*args, **kwargs):
        """
        
        __lt__( (boost_type_index)arg1, (boost_type_index)arg2) -> object :
        
            C++ signature :
                _object* __lt__(boost::typeindex::stl_type_index {lvalue},boost::typeindex::stl_type_index)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def hash_code(*args, **kwargs):
        """
        
        hash_code( (boost_type_index)self) -> int :
            Returns an unspecified value (here denoted by hash code) such that for all std::type_info objects referring to the same type, their hash code is the same.
        
            C++ signature :
                unsigned long hash_code(boost::typeindex::stl_type_index {lvalue})
        """
    @staticmethod
    def name(*args, **kwargs):
        """
        
        name( (boost_type_index)self) -> str :
            Returns an implementation defined null-terminated character string containing the name of the type. No guarantees are given; in particular, the returned string can be identical for several types and change between invocations of the same program.
        
            C++ signature :
                char const* name(boost::typeindex::stl_type_index {lvalue})
        """
    @staticmethod
    def pretty_name(*args, **kwargs):
        """
        
        pretty_name( (boost_type_index)self) -> str :
            Human readible name.
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > pretty_name(boost::typeindex::stl_type_index {lvalue})
        """
class std_type_index(Boost.Python.instance):
    """
    The class type_index holds implementation-specific information about a type, including the name of the type and means to compare two types for equality or collating order.
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (std_type_index)arg1, (std_type_index)arg2) -> object :
        
            C++ signature :
                _object* __eq__(std::type_index {lvalue},std::type_index)
        """
    @staticmethod
    def __ge__(*args, **kwargs):
        """
        
        __ge__( (std_type_index)arg1, (std_type_index)arg2) -> object :
        
            C++ signature :
                _object* __ge__(std::type_index {lvalue},std::type_index)
        """
    @staticmethod
    def __gt__(*args, **kwargs):
        """
        
        __gt__( (std_type_index)arg1, (std_type_index)arg2) -> object :
        
            C++ signature :
                _object* __gt__(std::type_index {lvalue},std::type_index)
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        Raises an exception
        This class cannot be instantiated from Python
        """
    @staticmethod
    def __le__(*args, **kwargs):
        """
        
        __le__( (std_type_index)arg1, (std_type_index)arg2) -> object :
        
            C++ signature :
                _object* __le__(std::type_index {lvalue},std::type_index)
        """
    @staticmethod
    def __lt__(*args, **kwargs):
        """
        
        __lt__( (std_type_index)arg1, (std_type_index)arg2) -> object :
        
            C++ signature :
                _object* __lt__(std::type_index {lvalue},std::type_index)
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def hash_code(*args, **kwargs):
        """
        
        hash_code( (std_type_index)self) -> int :
            Returns an unspecified value (here denoted by hash code) such that for all std::type_info objects referring to the same type, their hash code is the same.
        
            C++ signature :
                unsigned long hash_code(std::type_index {lvalue})
        """
    @staticmethod
    def name(*args, **kwargs):
        """
        
        name( (std_type_index)self) -> str :
            Returns an implementation defined null-terminated character string containing the name of the type. No guarantees are given; in particular, the returned string can be identical for several types and change between invocations of the same program.
        
            C++ signature :
                char const* name(std::type_index {lvalue})
        """
    @staticmethod
    def pretty_name(*args, **kwargs):
        """
        
        pretty_name( (std_type_index)self) -> str :
            Human readible name.
        
            C++ signature :
                std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > pretty_name(std::type_index)
        """
def SE3ToVector(*args, **kwargs):
    """
    
    SE3ToVector( (object)M, (numpy.ndarray)vec) -> None :
        Convert the input SE3 object M to a 12D vector of floats [X,Y,Z,R11,R12,R13,R14,...] vec
    
        C++ signature :
            void SE3ToVector(pinocchio::SE3Tpl<double, 0>,Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1>, 0, Eigen::InnerStride<1> >)
    
    SE3ToVector( (object)M) -> numpy.ndarray :
        Convert the input SE3 object M to a 12D vector of floats [X,Y,Z,R11,R12,R13,R14,...] and return the vector
    
        C++ signature :
            Eigen::Matrix<double, -1, 1, 0, -1, 1> SE3ToVector(pinocchio::SE3Tpl<double, 0>)
    """
def seed(*args, **kwargs):
    """
    
    seed( (int)seed_value) -> None :
        Initialize the pseudo-random number generator with the argument seed_value.
    
        C++ signature :
            void seed(unsigned int)
    """
def sharedMemory(*args, **kwargs):
    """
    
    sharedMemory( (bool)value) -> None :
        Share the memory when converting from Eigen to Numpy.
    
        C++ signature :
            void sharedMemory(bool)
    
    sharedMemory() -> bool :
        Status of the shared memory when converting from Eigen to Numpy.
        If True, the memory is shared when converting an Eigen::Matrix to a numpy.array.
        Otherwise, a deep copy of the Eigen::Matrix is performed.
    
        C++ signature :
            bool sharedMemory()
    """
def vectorToSE3(*args, **kwargs):
    """
    
    vectorToSE3( (numpy.ndarray)vec, (object)M) -> None :
        Convert the input 12D vector of floats [X,Y,Z,R11,R12,R13,R14,...] vec to a SE3 object M
    
        C++ signature :
            void vectorToSE3(Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1>, 0, Eigen::InnerStride<1> >,pinocchio::SE3Tpl<double, 0> {lvalue})
    
    vectorToSE3( (numpy.ndarray)vec) -> object :
        Convert the input 12D vector of floats [X,Y,Z,R11,R12,R13,R14,...] vec to a SE3 object and return the SE3 object
    
        C++ signature :
            pinocchio::SE3Tpl<double, 0> vectorToSE3(Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1>, 0, Eigen::InnerStride<1> >)
    """
FIXED_BASE_SYSTEM: RootJointType  # value = tsid.tsid_pywrap.RootJointType.FIXED_BASE_SYSTEM
FLOATING_BASE_SYSTEM: RootJointType  # value = tsid.tsid_pywrap.RootJointType.FLOATING_BASE_SYSTEM
