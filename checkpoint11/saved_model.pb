ЄЃ&
с¬
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ъ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Ј—8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
ј
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ъ∞"
В
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0
Г
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А* 
shared_nameconv2d_3/kernel
|
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*'
_output_shapes
:@А*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:А*
dtype0
П
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_1/gamma
И
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_1/beta
Ж
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_1/moving_mean
Ф
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:А*
dtype0
£
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_1/moving_variance
Ь
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:А*
dtype0
Г
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А* 
shared_nameconv2d_2/kernel
|
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:@А*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:А*
dtype0
Д
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv2d_5/kernel
}
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*(
_output_shapes
:АА*
dtype0
s
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_5/bias
l
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes	
:А*
dtype0
П
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_2/gamma
И
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_2/beta
Ж
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_2/moving_mean
Ф
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:А*
dtype0
£
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_2/moving_variance
Ь
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:А*
dtype0
Д
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv2d_4/kernel
}
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*(
_output_shapes
:АА*
dtype0
s
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_4/bias
l
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:А*
dtype0
Д
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv2d_7/kernel
}
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*(
_output_shapes
:АА*
dtype0
s
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_7/bias
l
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes	
:А*
dtype0
П
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_3/gamma
И
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_3/beta
Ж
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_3/moving_mean
Ф
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:А*
dtype0
£
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_3/moving_variance
Ь
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:А*
dtype0
Д
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv2d_6/kernel
}
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*(
_output_shapes
:АА*
dtype0
s
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_6/bias
l
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes	
:А*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
АА*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:А*
dtype0
n
p/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А∞!*
shared_name
p/kernel
g
p/kernel/Read/ReadVariableOpReadVariableOpp/kernel* 
_output_shapes
:
А∞!*
dtype0
e
p/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:∞!*
shared_namep/bias
^
p/bias/Read/ReadVariableOpReadVariableOpp/bias*
_output_shapes	
:∞!*
dtype0
m
v/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_name
v/kernel
f
v/kernel/Read/ReadVariableOpReadVariableOpv/kernel*
_output_shapes
:	А*
dtype0
d
v/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namev/bias
]
v/bias/Read/ReadVariableOpReadVariableOpv/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
Р
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_1/kernel/m
Й
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:@*
dtype0
А
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
Ш
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/m
С
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:@*
dtype0
Ц
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/m
П
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:@*
dtype0
М
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/m
Е
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:@*
dtype0
С
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*'
shared_nameAdam/conv2d_3/kernel/m
К
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*'
_output_shapes
:@А*
dtype0
Б
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_3/bias/m
z
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_1/gamma/m
Ц
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_1/beta/m
Ф
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes	
:А*
dtype0
С
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*'
shared_nameAdam/conv2d_2/kernel/m
К
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*'
_output_shapes
:@А*
dtype0
Б
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_2/bias/m
z
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_5/kernel/m
Л
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*(
_output_shapes
:АА*
dtype0
Б
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_5/bias/m
z
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_2/gamma/m
Ц
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_2/beta/m
Ф
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_4/kernel/m
Л
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*(
_output_shapes
:АА*
dtype0
Б
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_4/bias/m
z
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_7/kernel/m
Л
*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*(
_output_shapes
:АА*
dtype0
Б
Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_7/bias/m
z
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_3/gamma/m
Ц
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_3/beta/m
Ф
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_6/kernel/m
Л
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*(
_output_shapes
:АА*
dtype0
Б
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_6/bias/m
z
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes	
:А*
dtype0
Д
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
АА*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:А*
dtype0
|
Adam/p/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А∞!* 
shared_nameAdam/p/kernel/m
u
#Adam/p/kernel/m/Read/ReadVariableOpReadVariableOpAdam/p/kernel/m* 
_output_shapes
:
А∞!*
dtype0
s
Adam/p/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:∞!*
shared_nameAdam/p/bias/m
l
!Adam/p/bias/m/Read/ReadVariableOpReadVariableOpAdam/p/bias/m*
_output_shapes	
:∞!*
dtype0
{
Adam/v/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_nameAdam/v/kernel/m
t
#Adam/v/kernel/m/Read/ReadVariableOpReadVariableOpAdam/v/kernel/m*
_output_shapes
:	А*
dtype0
r
Adam/v/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/v/bias/m
k
!Adam/v/bias/m/Read/ReadVariableOpReadVariableOpAdam/v/bias/m*
_output_shapes
:*
dtype0
Р
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_1/kernel/v
Й
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:@*
dtype0
А
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
Ш
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/v
С
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:@*
dtype0
Ц
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/v
П
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:@*
dtype0
М
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/v
Е
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:@*
dtype0
С
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*'
shared_nameAdam/conv2d_3/kernel/v
К
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*'
_output_shapes
:@А*
dtype0
Б
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_3/bias/v
z
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_1/gamma/v
Ц
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_1/beta/v
Ф
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes	
:А*
dtype0
С
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*'
shared_nameAdam/conv2d_2/kernel/v
К
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*'
_output_shapes
:@А*
dtype0
Б
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_2/bias/v
z
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_5/kernel/v
Л
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*(
_output_shapes
:АА*
dtype0
Б
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_5/bias/v
z
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_2/gamma/v
Ц
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_2/beta/v
Ф
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_4/kernel/v
Л
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*(
_output_shapes
:АА*
dtype0
Б
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_4/bias/v
z
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_7/kernel/v
Л
*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*(
_output_shapes
:АА*
dtype0
Б
Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_7/bias/v
z
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_3/gamma/v
Ц
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes	
:А*
dtype0
Ы
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/batch_normalization_3/beta/v
Ф
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes	
:А*
dtype0
Т
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv2d_6/kernel/v
Л
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*(
_output_shapes
:АА*
dtype0
Б
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2d_6/bias/v
z
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes	
:А*
dtype0
Д
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
АА*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:А*
dtype0
|
Adam/p/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А∞!* 
shared_nameAdam/p/kernel/v
u
#Adam/p/kernel/v/Read/ReadVariableOpReadVariableOpAdam/p/kernel/v* 
_output_shapes
:
А∞!*
dtype0
s
Adam/p/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:∞!*
shared_nameAdam/p/bias/v
l
!Adam/p/bias/v/Read/ReadVariableOpReadVariableOpAdam/p/bias/v*
_output_shapes	
:∞!*
dtype0
{
Adam/v/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_nameAdam/v/kernel/v
t
#Adam/v/kernel/v/Read/ReadVariableOpReadVariableOpAdam/v/kernel/v*
_output_shapes
:	А*
dtype0
r
Adam/v/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/v/bias/v
k
!Adam/v/bias/v/Read/ReadVariableOpReadVariableOpAdam/v/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ех
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ях
valueФхBРх BИх
Д
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer-17
layer-18
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
layer_with_weights-11
layer-21
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
layer_with_weights-14
layer-27
	optimizer
loss

signatures
# _self_saveable_object_factories
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_default_save_signature*
'
#(_self_saveable_object_factories* 
Ћ

)kernel
*bias
#+_self_saveable_object_factories
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses*
ъ
2axis
	3gamma
4beta
5moving_mean
6moving_variance
#7_self_saveable_object_factories
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses*
Ћ

>kernel
?bias
#@_self_saveable_object_factories
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
≥
#G_self_saveable_object_factories
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses* 
≥
#N_self_saveable_object_factories
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses* 
≥
#U_self_saveable_object_factories
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses* 
Ћ

\kernel
]bias
#^_self_saveable_object_factories
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses*
ъ
eaxis
	fgamma
gbeta
hmoving_mean
imoving_variance
#j_self_saveable_object_factories
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses*
Ћ

qkernel
rbias
#s_self_saveable_object_factories
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses*
і
#z_self_saveable_object_factories
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+А&call_and_return_all_conditional_losses* 
Ї
$Б_self_saveable_object_factories
В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses* 
Ї
$И_self_saveable_object_factories
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
Н__call__
+О&call_and_return_all_conditional_losses* 
‘
Пkernel
	Рbias
$С_self_saveable_object_factories
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses*
Ж
	Шaxis

Щgamma
	Ъbeta
Ыmoving_mean
Ьmoving_variance
$Э_self_saveable_object_factories
Ю	variables
Яtrainable_variables
†regularization_losses
°	keras_api
Ґ__call__
+£&call_and_return_all_conditional_losses*
‘
§kernel
	•bias
$¶_self_saveable_object_factories
І	variables
®trainable_variables
©regularization_losses
™	keras_api
Ђ__call__
+ђ&call_and_return_all_conditional_losses*
Ї
$≠_self_saveable_object_factories
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
±	keras_api
≤__call__
+≥&call_and_return_all_conditional_losses* 
Ї
$і_self_saveable_object_factories
µ	variables
ґtrainable_variables
Јregularization_losses
Є	keras_api
є__call__
+Ї&call_and_return_all_conditional_losses* 
Ї
$ї_self_saveable_object_factories
Љ	variables
љtrainable_variables
Њregularization_losses
њ	keras_api
ј__call__
+Ѕ&call_and_return_all_conditional_losses* 
‘
¬kernel
	√bias
$ƒ_self_saveable_object_factories
≈	variables
∆trainable_variables
«regularization_losses
»	keras_api
…__call__
+ &call_and_return_all_conditional_losses*
Ж
	Ћaxis

ћgamma
	Ќbeta
ќmoving_mean
ѕmoving_variance
$–_self_saveable_object_factories
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
’__call__
+÷&call_and_return_all_conditional_losses*
‘
„kernel
	Ўbias
$ў_self_saveable_object_factories
Џ	variables
џtrainable_variables
№regularization_losses
Ё	keras_api
ё__call__
+я&call_and_return_all_conditional_losses*
Ї
$а_self_saveable_object_factories
б	variables
вtrainable_variables
гregularization_losses
д	keras_api
е__call__
+ж&call_and_return_all_conditional_losses* 
Ї
$з_self_saveable_object_factories
и	variables
йtrainable_variables
кregularization_losses
л	keras_api
м__call__
+н&call_and_return_all_conditional_losses* 
Ї
$о_self_saveable_object_factories
п	variables
рtrainable_variables
сregularization_losses
т	keras_api
у__call__
+ф&call_and_return_all_conditional_losses* 
‘
хkernel
	цbias
$ч_self_saveable_object_factories
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses*
‘
юkernel
	€bias
$А_self_saveable_object_factories
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses*
‘
Зkernel
	Иbias
$Й_self_saveable_object_factories
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses*
≈
	Рiter
Сbeta_1
Тbeta_2

Уdecay
Фlearning_rate)m±*m≤3m≥4mі>mµ?mґ\mЈ]mЄfmєgmЇqmїrmЉ	Пmљ	РmЊ	Щmњ	Ъmј	§mЅ	•m¬	¬m√	√mƒ	ћm≈	Ќm∆	„m«	Ўm»	хm…	цm 	юmЋ	€mћ	ЗmЌ	Иmќ)vѕ*v–3v—4v“>v”?v‘\v’]v÷fv„gvЎqvўrvЏ	Пvџ	Рv№	ЩvЁ	Ъvё	§vя	•vа	¬vб	√vв	ћvг	Ќvд	„vе	Ўvж	хvз	цvи	юvй	€vк	Зvл	Иvм*
* 

Хserving_default* 
* 
ј
)0
*1
32
43
54
65
>6
?7
\8
]9
f10
g11
h12
i13
q14
r15
П16
Р17
Щ18
Ъ19
Ы20
Ь21
§22
•23
¬24
√25
ћ26
Ќ27
ќ28
ѕ29
„30
Ў31
х32
ц33
ю34
€35
З36
И37*
ь
)0
*1
32
43
>4
?5
\6
]7
f8
g9
q10
r11
П12
Р13
Щ14
Ъ15
§16
•17
¬18
√19
ћ20
Ќ21
„22
Ў23
х24
ц25
ю26
€27
З28
И29*
* 
µ
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
'_default_save_signature
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
* 
* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

)0
*1*

)0
*1*
* 
Ш
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*
* 
* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
30
41
52
63*

30
41*
* 
Ш
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

>0
?1*

>0
?1*
* 
Ш
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ц
™non_trainable_variables
Ђlayers
ђmetrics
 ≠layer_regularization_losses
Ѓlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ц
ѓnon_trainable_variables
∞layers
±metrics
 ≤layer_regularization_losses
≥layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ц
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

\0
]1*

\0
]1*
* 
Ш
єnon_trainable_variables
Їlayers
їmetrics
 Љlayer_regularization_losses
љlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
f0
g1
h2
i3*

f0
g1*
* 
Ш
Њnon_trainable_variables
њlayers
јmetrics
 Ѕlayer_regularization_losses
¬layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

q0
r1*

q0
r1*
* 
Ш
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ш
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ь
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ь
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
Й	variables
Кtrainable_variables
Лregularization_losses
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

П0
Р1*

П0
Р1*
* 
Ю
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
Щ0
Ъ1
Ы2
Ь3*

Щ0
Ъ1*
* 
Ю
№non_trainable_variables
Ёlayers
ёmetrics
 яlayer_regularization_losses
аlayer_metrics
Ю	variables
Яtrainable_variables
†regularization_losses
Ґ__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

§0
•1*

§0
•1*
* 
Ю
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
І	variables
®trainable_variables
©regularization_losses
Ђ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
≤__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ь
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
µ	variables
ґtrainable_variables
Јregularization_losses
є__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ь
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
Љ	variables
љtrainable_variables
Њregularization_losses
ј__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

¬0
√1*

¬0
√1*
* 
Ю
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
≈	variables
∆trainable_variables
«regularization_losses
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_3/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_3/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_3/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_3/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
ћ0
Ќ1
ќ2
ѕ3*

ћ0
Ќ1*
* 
Ю
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
—	variables
“trainable_variables
”regularization_losses
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEconv2d_6/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_6/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

„0
Ў1*

„0
Ў1*
* 
Ю
€non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
Џ	variables
џtrainable_variables
№regularization_losses
ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
б	variables
вtrainable_variables
гregularization_losses
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ь
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
и	variables
йtrainable_variables
кregularization_losses
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ь
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
п	variables
рtrainable_variables
сregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUEdense/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
dense/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

х0
ц1*

х0
ц1*
* 
Ю
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUEp/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEp/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

ю0
€1*

ю0
€1*
* 
Ю
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUEv/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEv/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

З0
И1*

З0
И1*
* 
Ю
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
@
50
61
h2
i3
Ы4
Ь5
ќ6
ѕ7*
Џ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27*

Ґ0
£1
§2*
* 
* 
* 
* 
* 
* 
* 

50
61*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

h0
i1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ы0
Ь1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ќ0
ѕ1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

•total

¶count
І	variables
®	keras_api*
<

©total

™count
Ђ	variables
ђ	keras_api*
<

≠total

Ѓcount
ѓ	variables
∞	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

•0
¶1*

І	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

©0
™1*

Ђ	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*

≠0
Ѓ1*

ѓ	variables*
В|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_7/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_7/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/conv2d_6/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_6/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/dense/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/p/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/p/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/v/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/v/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_7/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_7/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Г}
VARIABLE_VALUEAdam/conv2d_6/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_6/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/dense/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dense/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/p/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/p/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/v/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/v/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Т
serving_default_input_1Placeholder*3
_output_shapes!
:€€€€€€€€€*
dtype0*(
shape:€€€€€€€€€
“	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_1/kernelconv2d_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d/kernelconv2d/biasconv2d_3/kernelconv2d_3/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasconv2d_5/kernelconv2d_5/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_4/kernelconv2d_4/biasconv2d_7/kernelconv2d_7/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_6/kernelconv2d_6/biasdense/kernel
dense/biasv/kernelv/biasp/kernelp/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':€€€€€€€€€∞!:€€€€€€€€€*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_3339582
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ъ'
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpp/kernel/Read/ReadVariableOpp/bias/Read/ReadVariableOpv/kernel/Read/ReadVariableOpv/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_7/kernel/m/Read/ReadVariableOp(Adam/conv2d_7/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp#Adam/p/kernel/m/Read/ReadVariableOp!Adam/p/bias/m/Read/ReadVariableOp#Adam/v/kernel/m/Read/ReadVariableOp!Adam/v/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_7/kernel/v/Read/ReadVariableOp(Adam/conv2d_7/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp#Adam/p/kernel/v/Read/ReadVariableOp!Adam/p/bias/v/Read/ReadVariableOp#Adam/v/kernel/v/Read/ReadVariableOp!Adam/v/bias/v/Read/ReadVariableOpConst*z
Tins
q2o	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_save_3340705
С
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_1/kernelconv2d_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d/kernelconv2d/biasconv2d_3/kernelconv2d_3/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasconv2d_5/kernelconv2d_5/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_4/kernelconv2d_4/biasconv2d_7/kernelconv2d_7/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_6/kernelconv2d_6/biasdense/kernel
dense/biasp/kernelp/biasv/kernelv/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/conv2d_1/kernel/mAdam/conv2d_1/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/p/kernel/mAdam/p/bias/mAdam/v/kernel/mAdam/v/bias/mAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/p/kernel/vAdam/p/bias/vAdam/v/kernel/vAdam/v/bias/v*y
Tinr
p2n*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__traced_restore_3341042ґТ
Ќ	
÷
7__inference_batch_normalization_3_layer_call_fn_3340184

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3337362Ч
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
м
S
'__inference_add_3_layer_call_fn_3340268
inputs_0
inputs_1
identity«
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_3_layer_call_and_return_conditional_losses_3337782m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€А:^ Z
4
_output_shapes"
 :€€€€€€€€€А
"
_user_specified_name
inputs/0:^Z
4
_output_shapes"
 :€€€€€€€€€А
"
_user_specified_name
inputs/1
Є
≈
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3340042

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0А
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
П

р
>__inference_v_layer_call_and_return_conditional_losses_3337826

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ѓ%
і
E__inference_conv2d_3_layer_call_and_return_conditional_losses_3339802

inputs@
%conv2d_conv2d_readvariableop_resource:@АA
2squeeze_batch_dims_biasadd_readvariableop_resource:	А
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Л
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:К
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аa
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   ¶
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЩ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Є
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аw
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АС
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
Є
≈
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3339864

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0А
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
о
M
1__inference_max_pooling3d_1_layer_call_fn_3339933

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_3337230Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Б
e
I__inference_activation_3_layer_call_and_return_conditional_losses_3340284

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :€€€€€€€€€Аg
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
Ґ%
∞
C__inference_conv2d_layer_call_and_return_conditional_losses_3339728

inputs?
%conv2d_conv2d_readvariableop_resource:@@
2squeeze_batch_dims_biasadd_readvariableop_resource:@
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   •
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Ш
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ј
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@q
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€@С
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
юy
П
B__inference_model_layer_call_and_return_conditional_losses_3338673
input_1*
conv2d_1_3338568:@
conv2d_1_3338570:@)
batch_normalization_3338573:@)
batch_normalization_3338575:@)
batch_normalization_3338577:@)
batch_normalization_3338579:@(
conv2d_3338582:@
conv2d_3338584:@+
conv2d_3_3338590:@А
conv2d_3_3338592:	А,
batch_normalization_1_3338595:	А,
batch_normalization_1_3338597:	А,
batch_normalization_1_3338599:	А,
batch_normalization_1_3338601:	А+
conv2d_2_3338604:@А
conv2d_2_3338606:	А,
conv2d_5_3338612:АА
conv2d_5_3338614:	А,
batch_normalization_2_3338617:	А,
batch_normalization_2_3338619:	А,
batch_normalization_2_3338621:	А,
batch_normalization_2_3338623:	А,
conv2d_4_3338626:АА
conv2d_4_3338628:	А,
conv2d_7_3338634:АА
conv2d_7_3338636:	А,
batch_normalization_3_3338639:	А,
batch_normalization_3_3338641:	А,
batch_normalization_3_3338643:	А,
batch_normalization_3_3338645:	А,
conv2d_6_3338648:АА
conv2d_6_3338650:	А!
dense_3338656:
АА
dense_3338658:	А
	v_3338661:	А
	v_3338663:
	p_3338666:
А∞!
	p_3338668:	∞!
identity

identity_1ИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐp/StatefulPartitionedCallҐv/StatefulPartitionedCallА
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_1_3338568conv2d_1_3338570*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3337413К
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_3338573batch_normalization_3338575batch_normalization_3338577batch_normalization_3338579*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3337134ш
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_3338582conv2d_3338584*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_3337461У
add/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_3337473я
activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_3337480м
max_pooling3d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_3337154†
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv2d_3_3338590conv2d_3_3338592*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_3337516Ч
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_1_3338595batch_normalization_1_3338597batch_normalization_1_3338599batch_normalization_1_3338601*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3337210†
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv2d_2_3338604conv2d_2_3338606*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_3337564Ь
add_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_3337576ж
activation_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_3337583у
max_pooling3d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_3337230Ґ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv2d_5_3338612conv2d_5_3338614*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3337619Ч
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_2_3338617batch_normalization_2_3338619batch_normalization_2_3338621batch_normalization_2_3338623*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3337286Ґ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv2d_4_3338626conv2d_4_3338628*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_3337667Ь
add_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_3337679ж
activation_2/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_3337686у
max_pooling3d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_3337306Ґ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv2d_7_3338634conv2d_7_3338636*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3337722Ч
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_3_3338639batch_normalization_3_3338641batch_normalization_3_3338643batch_normalization_3_3338645*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3337362Ґ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv2d_6_3338648conv2d_6_3338650*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3337770Ь
add_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_3_layer_call_and_return_conditional_losses_3337782ж
activation_3/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_3337789„
flatten/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3337797В
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_3338656dense_3338658*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3337809ч
v/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0	v_3338661	v_3338663*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_v_layer_call_and_return_conditional_losses_3337826т
p/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	p_3338666	p_3338668*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€∞!*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_p_layer_call_and_return_conditional_losses_3337843r
IdentityIdentity"p/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞!s

Identity_1Identity"v/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€т
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall^p/StatefulPartitionedCall^v/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall26
p/StatefulPartitionedCallp/StatefulPartitionedCall26
v/StatefulPartitionedCallv/StatefulPartitionedCall:\ X
3
_output_shapes!
:€€€€€€€€€
!
_user_specified_name	input_1
є
С
#__inference_v_layer_call_fn_3340343

inputs
unknown:	А
	unknown_0:
identityИҐStatefulPartitionedCall”
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_v_layer_call_and_return_conditional_losses_3337826o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ќ	
÷
7__inference_batch_normalization_2_layer_call_fn_3340006

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3337286Ч
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
¶
њ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3337134

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
–	
ц
B__inference_dense_layer_call_and_return_conditional_losses_3340314

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ѓ%
і
E__inference_conv2d_2_layer_call_and_return_conditional_losses_3337564

inputs@
%conv2d_conv2d_readvariableop_resource:@АA
2squeeze_batch_dims_biasadd_readvariableop_resource:	А
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Л
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:К
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аa
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   ¶
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЩ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Є
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аw
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АС
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
Жz
П
B__inference_model_layer_call_and_return_conditional_losses_3338565
input_1*
conv2d_1_3338460:@
conv2d_1_3338462:@)
batch_normalization_3338465:@)
batch_normalization_3338467:@)
batch_normalization_3338469:@)
batch_normalization_3338471:@(
conv2d_3338474:@
conv2d_3338476:@+
conv2d_3_3338482:@А
conv2d_3_3338484:	А,
batch_normalization_1_3338487:	А,
batch_normalization_1_3338489:	А,
batch_normalization_1_3338491:	А,
batch_normalization_1_3338493:	А+
conv2d_2_3338496:@А
conv2d_2_3338498:	А,
conv2d_5_3338504:АА
conv2d_5_3338506:	А,
batch_normalization_2_3338509:	А,
batch_normalization_2_3338511:	А,
batch_normalization_2_3338513:	А,
batch_normalization_2_3338515:	А,
conv2d_4_3338518:АА
conv2d_4_3338520:	А,
conv2d_7_3338526:АА
conv2d_7_3338528:	А,
batch_normalization_3_3338531:	А,
batch_normalization_3_3338533:	А,
batch_normalization_3_3338535:	А,
batch_normalization_3_3338537:	А,
conv2d_6_3338540:АА
conv2d_6_3338542:	А!
dense_3338548:
АА
dense_3338550:	А
	v_3338553:	А
	v_3338555:
	p_3338558:
А∞!
	p_3338560:	∞!
identity

identity_1ИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐp/StatefulPartitionedCallҐv/StatefulPartitionedCallА
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_1_3338460conv2d_1_3338462*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3337413М
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_3338465batch_normalization_3338467batch_normalization_3338469batch_normalization_3338471*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3337103ш
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_3338474conv2d_3338476*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_3337461У
add/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_3337473я
activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_3337480м
max_pooling3d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_3337154†
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv2d_3_3338482conv2d_3_3338484*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_3337516Щ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_1_3338487batch_normalization_1_3338489batch_normalization_1_3338491batch_normalization_1_3338493*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3337179†
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv2d_2_3338496conv2d_2_3338498*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_3337564Ь
add_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_3337576ж
activation_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_3337583у
max_pooling3d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_3337230Ґ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv2d_5_3338504conv2d_5_3338506*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3337619Щ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_2_3338509batch_normalization_2_3338511batch_normalization_2_3338513batch_normalization_2_3338515*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3337255Ґ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv2d_4_3338518conv2d_4_3338520*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_3337667Ь
add_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_3337679ж
activation_2/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_3337686у
max_pooling3d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_3337306Ґ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv2d_7_3338526conv2d_7_3338528*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3337722Щ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_3_3338531batch_normalization_3_3338533batch_normalization_3_3338535batch_normalization_3_3338537*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3337331Ґ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv2d_6_3338540conv2d_6_3338542*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3337770Ь
add_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_3_layer_call_and_return_conditional_losses_3337782ж
activation_3/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_3337789„
flatten/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3337797В
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_3338548dense_3338550*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3337809ч
v/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0	v_3338553	v_3338555*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_v_layer_call_and_return_conditional_losses_3337826т
p/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	p_3338558	p_3338560*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€∞!*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_p_layer_call_and_return_conditional_losses_3337843r
IdentityIdentity"p/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞!s

Identity_1Identity"v/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€т
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall^p/StatefulPartitionedCall^v/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall26
p/StatefulPartitionedCallp/StatefulPartitionedCall26
v/StatefulPartitionedCallv/StatefulPartitionedCall:\ X
3
_output_shapes!
:€€€€€€€€€
!
_user_specified_name	input_1
о
ѕ	
'__inference_model_layer_call_fn_3338457
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@
	unknown_6:@$
	unknown_7:@А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А%

unknown_13:@А

unknown_14:	А&

unknown_15:АА

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А&

unknown_29:АА

unknown_30:	А

unknown_31:
АА

unknown_32:	А

unknown_33:	А

unknown_34:

unknown_35:
А∞!

unknown_36:	∞!
identity

identity_1ИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':€€€€€€€€€∞!:€€€€€€€€€*@
_read_only_resource_inputs"
 	
 !"#$%&*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_3338293p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞!q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:€€€€€€€€€
!
_user_specified_name	input_1
А
°
*__inference_conv2d_3_layer_call_fn_3339769

inputs"
unknown:@А
	unknown_0:	А
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_3337516|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
•Я
Џ"
B__inference_model_layer_call_and_return_conditional_losses_3339171

inputsH
.conv2d_1_conv2d_conv2d_readvariableop_resource:@I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@F
,conv2d_conv2d_conv2d_readvariableop_resource:@G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:@I
.conv2d_3_conv2d_conv2d_readvariableop_resource:@АJ
;conv2d_3_squeeze_batch_dims_biasadd_readvariableop_resource:	А<
-batch_normalization_1_readvariableop_resource:	А>
/batch_normalization_1_readvariableop_1_resource:	АM
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	АI
.conv2d_2_conv2d_conv2d_readvariableop_resource:@АJ
;conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource:	АJ
.conv2d_5_conv2d_conv2d_readvariableop_resource:ААJ
;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:	А<
-batch_normalization_2_readvariableop_resource:	А>
/batch_normalization_2_readvariableop_1_resource:	АM
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	АJ
.conv2d_4_conv2d_conv2d_readvariableop_resource:ААJ
;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:	АJ
.conv2d_7_conv2d_conv2d_readvariableop_resource:ААJ
;conv2d_7_squeeze_batch_dims_biasadd_readvariableop_resource:	А<
-batch_normalization_3_readvariableop_resource:	А>
/batch_normalization_3_readvariableop_1_resource:	АM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	АJ
.conv2d_6_conv2d_conv2d_readvariableop_resource:ААJ
;conv2d_6_squeeze_batch_dims_biasadd_readvariableop_resource:	А8
$dense_matmul_readvariableop_resource:
АА4
%dense_biasadd_readvariableop_resource:	А3
 v_matmul_readvariableop_resource:	А/
!v_biasadd_readvariableop_resource:4
 p_matmul_readvariableop_resource:
А∞!0
!p_biasadd_readvariableop_resource:	∞!
identity

identity_1ИҐ3batch_normalization/FusedBatchNormV3/ReadVariableOpҐ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ"batch_normalization/ReadVariableOpҐ$batch_normalization/ReadVariableOp_1Ґ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_1/ReadVariableOpҐ&batch_normalization_1/ReadVariableOp_1Ґ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґ5batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_3/ReadVariableOpҐ&batch_normalization_3/ReadVariableOp_1Ґ#conv2d/Conv2D/Conv2D/ReadVariableOpҐ0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_1/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_2/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_3/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_4/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_5/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_6/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_7/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐp/BiasAdd/ReadVariableOpҐp/MatMul/ReadVariableOpҐv/BiasAdd/ReadVariableOpҐv/MatMul/ReadVariableOpK
conv2d_1/Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         М
conv2d_1/Conv2D/ReshapeReshapeinputs&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€Ь
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0”
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:§
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   ј
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@™
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0“
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@А
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:…
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@Г
conv2d_1/ReluRelu.conv2d_1/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@К
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0О
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0ђ
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0∞
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ћ
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
is_training( I
conv2d/Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Х
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         И
conv2d/Conv2D/ReshapeReshapeinputs$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€Ш
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ќ
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ј
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Ю
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: В
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskА
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   Ї
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@¶
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€р
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:√
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@
conv2d/ReluRelu,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@У
add/addAddV2(batch_normalization/FusedBatchNormV3:y:0conv2d/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€@b
activation/ReluReluadd/add:z:0*
T0*3
_output_shapes!
:€€€€€€€€€@њ
max_pooling3d/MaxPool3D	MaxPool3Dactivation/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€@*
ksize	
*
paddingVALID*
strides	
e
conv2d_3/Conv2D/ShapeShape max_pooling3d/MaxPool3D:output:0*
T0*
_output_shapes
:m
#conv2d_3/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_3/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_3/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_3/Conv2D/strided_sliceStridedSliceconv2d_3/Conv2D/Shape:output:0,conv2d_3/Conv2D/strided_slice/stack:output:0.conv2d_3/Conv2D/strided_slice/stack_1:output:0.conv2d_3/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_3/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   ¶
conv2d_3/Conv2D/ReshapeReshape max_pooling3d/MaxPool3D:output:0&conv2d_3/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Э
%conv2d_3/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0‘
conv2d_3/Conv2D/Conv2DConv2D conv2d_3/Conv2D/Reshape:output:0-conv2d_3/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
t
conv2d_3/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   f
conv2d_3/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_3/Conv2D/concatConcatV2&conv2d_3/Conv2D/strided_slice:output:0(conv2d_3/Conv2D/concat/values_1:output:0$conv2d_3/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:•
conv2d_3/Conv2D/Reshape_1Reshapeconv2d_3/Conv2D/Conv2D:output:0conv2d_3/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
!conv2d_3/squeeze_batch_dims/ShapeShape"conv2d_3/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_3/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_3/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_3/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_3/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_3/squeeze_batch_dims/Shape:output:08conv2d_3/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_3/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_3/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_3/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   Ѕ
#conv2d_3/squeeze_batch_dims/ReshapeReshape"conv2d_3/Conv2D/Reshape_1:output:02conv2d_3/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_3_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0”
#conv2d_3/squeeze_batch_dims/BiasAddBiasAdd,conv2d_3/squeeze_batch_dims/Reshape:output:0:conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АА
+conv2d_3/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   r
'conv2d_3/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_3/squeeze_batch_dims/concatConcatV22conv2d_3/squeeze_batch_dims/strided_slice:output:04conv2d_3/squeeze_batch_dims/concat/values_1:output:00conv2d_3/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
: 
%conv2d_3/squeeze_batch_dims/Reshape_1Reshape,conv2d_3/squeeze_batch_dims/BiasAdd:output:0+conv2d_3/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АД
conv2d_3/ReluRelu.conv2d_3/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АП
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Џ
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( e
conv2d_2/Conv2D/ShapeShape max_pooling3d/MaxPool3D:output:0*
T0*
_output_shapes
:m
#conv2d_2/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_2/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_2/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_2/Conv2D/strided_sliceStridedSliceconv2d_2/Conv2D/Shape:output:0,conv2d_2/Conv2D/strided_slice/stack:output:0.conv2d_2/Conv2D/strided_slice/stack_1:output:0.conv2d_2/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_2/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   ¶
conv2d_2/Conv2D/ReshapeReshape max_pooling3d/MaxPool3D:output:0&conv2d_2/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Э
%conv2d_2/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0‘
conv2d_2/Conv2D/Conv2DConv2D conv2d_2/Conv2D/Reshape:output:0-conv2d_2/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
t
conv2d_2/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   f
conv2d_2/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_2/Conv2D/concatConcatV2&conv2d_2/Conv2D/strided_slice:output:0(conv2d_2/Conv2D/concat/values_1:output:0$conv2d_2/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:•
conv2d_2/Conv2D/Reshape_1Reshapeconv2d_2/Conv2D/Conv2D:output:0conv2d_2/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
!conv2d_2/squeeze_batch_dims/ShapeShape"conv2d_2/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_2/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_2/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_2/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_2/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_2/squeeze_batch_dims/Shape:output:08conv2d_2/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_2/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_2/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_2/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   Ѕ
#conv2d_2/squeeze_batch_dims/ReshapeReshape"conv2d_2/Conv2D/Reshape_1:output:02conv2d_2/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0”
#conv2d_2/squeeze_batch_dims/BiasAddBiasAdd,conv2d_2/squeeze_batch_dims/Reshape:output:0:conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АА
+conv2d_2/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   r
'conv2d_2/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_2/squeeze_batch_dims/concatConcatV22conv2d_2/squeeze_batch_dims/strided_slice:output:04conv2d_2/squeeze_batch_dims/concat/values_1:output:00conv2d_2/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
: 
%conv2d_2/squeeze_batch_dims/Reshape_1Reshape,conv2d_2/squeeze_batch_dims/BiasAdd:output:0+conv2d_2/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АД
conv2d_2/ReluRelu.conv2d_2/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АЪ
	add_1/addAddV2*batch_normalization_1/FusedBatchNormV3:y:0conv2d_2/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аg
activation_1/ReluReluadd_1/add:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аƒ
max_pooling3d_1/MaxPool3D	MaxPool3Dactivation_1/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
ksize	
*
paddingVALID*
strides	
g
conv2d_5/Conv2D/ShapeShape"max_pooling3d_1/MaxPool3D:output:0*
T0*
_output_shapes
:m
#conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_5/Conv2D/strided_sliceStridedSliceconv2d_5/Conv2D/Shape:output:0,conv2d_5/Conv2D/strided_slice/stack:output:0.conv2d_5/Conv2D/strided_slice/stack_1:output:0.conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   ©
conv2d_5/Conv2D/ReshapeReshape"max_pooling3d_1/MaxPool3D:output:0&conv2d_5/Conv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЮ
%conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0‘
conv2d_5/Conv2D/Conv2DConv2D conv2d_5/Conv2D/Reshape:output:0-conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
t
conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_5/Conv2D/concatConcatV2&conv2d_5/Conv2D/strided_slice:output:0(conv2d_5/Conv2D/concat/values_1:output:0$conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:•
conv2d_5/Conv2D/Reshape_1Reshapeconv2d_5/Conv2D/Conv2D:output:0conv2d_5/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
!conv2d_5/squeeze_batch_dims/ShapeShape"conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_5/squeeze_batch_dims/Shape:output:08conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Ѕ
#conv2d_5/squeeze_batch_dims/ReshapeReshape"conv2d_5/Conv2D/Reshape_1:output:02conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0”
#conv2d_5/squeeze_batch_dims/BiasAddBiasAdd,conv2d_5/squeeze_batch_dims/Reshape:output:0:conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АА
+conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_5/squeeze_batch_dims/concatConcatV22conv2d_5/squeeze_batch_dims/strided_slice:output:04conv2d_5/squeeze_batch_dims/concat/values_1:output:00conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
: 
%conv2d_5/squeeze_batch_dims/Reshape_1Reshape,conv2d_5/squeeze_batch_dims/BiasAdd:output:0+conv2d_5/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АД
conv2d_5/ReluRelu.conv2d_5/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АП
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Џ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_5/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( g
conv2d_4/Conv2D/ShapeShape"max_pooling3d_1/MaxPool3D:output:0*
T0*
_output_shapes
:m
#conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_4/Conv2D/strided_sliceStridedSliceconv2d_4/Conv2D/Shape:output:0,conv2d_4/Conv2D/strided_slice/stack:output:0.conv2d_4/Conv2D/strided_slice/stack_1:output:0.conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   ©
conv2d_4/Conv2D/ReshapeReshape"max_pooling3d_1/MaxPool3D:output:0&conv2d_4/Conv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЮ
%conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0‘
conv2d_4/Conv2D/Conv2DConv2D conv2d_4/Conv2D/Reshape:output:0-conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
t
conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_4/Conv2D/concatConcatV2&conv2d_4/Conv2D/strided_slice:output:0(conv2d_4/Conv2D/concat/values_1:output:0$conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:•
conv2d_4/Conv2D/Reshape_1Reshapeconv2d_4/Conv2D/Conv2D:output:0conv2d_4/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
!conv2d_4/squeeze_batch_dims/ShapeShape"conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_4/squeeze_batch_dims/Shape:output:08conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Ѕ
#conv2d_4/squeeze_batch_dims/ReshapeReshape"conv2d_4/Conv2D/Reshape_1:output:02conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0”
#conv2d_4/squeeze_batch_dims/BiasAddBiasAdd,conv2d_4/squeeze_batch_dims/Reshape:output:0:conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АА
+conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_4/squeeze_batch_dims/concatConcatV22conv2d_4/squeeze_batch_dims/strided_slice:output:04conv2d_4/squeeze_batch_dims/concat/values_1:output:00conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
: 
%conv2d_4/squeeze_batch_dims/Reshape_1Reshape,conv2d_4/squeeze_batch_dims/BiasAdd:output:0+conv2d_4/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АД
conv2d_4/ReluRelu.conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АЪ
	add_2/addAddV2*batch_normalization_2/FusedBatchNormV3:y:0conv2d_4/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аg
activation_2/ReluReluadd_2/add:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аƒ
max_pooling3d_2/MaxPool3D	MaxPool3Dactivation_2/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
ksize	
*
paddingVALID*
strides	
g
conv2d_7/Conv2D/ShapeShape"max_pooling3d_2/MaxPool3D:output:0*
T0*
_output_shapes
:m
#conv2d_7/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_7/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_7/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_7/Conv2D/strided_sliceStridedSliceconv2d_7/Conv2D/Shape:output:0,conv2d_7/Conv2D/strided_slice/stack:output:0.conv2d_7/Conv2D/strided_slice/stack_1:output:0.conv2d_7/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_7/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ©
conv2d_7/Conv2D/ReshapeReshape"max_pooling3d_2/MaxPool3D:output:0&conv2d_7/Conv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЮ
%conv2d_7/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_7_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0‘
conv2d_7/Conv2D/Conv2DConv2D conv2d_7/Conv2D/Reshape:output:0-conv2d_7/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
t
conv2d_7/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_7/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_7/Conv2D/concatConcatV2&conv2d_7/Conv2D/strided_slice:output:0(conv2d_7/Conv2D/concat/values_1:output:0$conv2d_7/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:•
conv2d_7/Conv2D/Reshape_1Reshapeconv2d_7/Conv2D/Conv2D:output:0conv2d_7/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
!conv2d_7/squeeze_batch_dims/ShapeShape"conv2d_7/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_7/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_7/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_7/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_7/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_7/squeeze_batch_dims/Shape:output:08conv2d_7/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_7/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_7/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_7/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Ѕ
#conv2d_7/squeeze_batch_dims/ReshapeReshape"conv2d_7/Conv2D/Reshape_1:output:02conv2d_7/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
2conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_7_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0”
#conv2d_7/squeeze_batch_dims/BiasAddBiasAdd,conv2d_7/squeeze_batch_dims/Reshape:output:0:conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АА
+conv2d_7/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_7/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_7/squeeze_batch_dims/concatConcatV22conv2d_7/squeeze_batch_dims/strided_slice:output:04conv2d_7/squeeze_batch_dims/concat/values_1:output:00conv2d_7/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
: 
%conv2d_7/squeeze_batch_dims/Reshape_1Reshape,conv2d_7/squeeze_batch_dims/BiasAdd:output:0+conv2d_7/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АД
conv2d_7/ReluRelu.conv2d_7/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АП
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Џ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( g
conv2d_6/Conv2D/ShapeShape"max_pooling3d_2/MaxPool3D:output:0*
T0*
_output_shapes
:m
#conv2d_6/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_6/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_6/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_6/Conv2D/strided_sliceStridedSliceconv2d_6/Conv2D/Shape:output:0,conv2d_6/Conv2D/strided_slice/stack:output:0.conv2d_6/Conv2D/strided_slice/stack_1:output:0.conv2d_6/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_6/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ©
conv2d_6/Conv2D/ReshapeReshape"max_pooling3d_2/MaxPool3D:output:0&conv2d_6/Conv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЮ
%conv2d_6/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_6_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0‘
conv2d_6/Conv2D/Conv2DConv2D conv2d_6/Conv2D/Reshape:output:0-conv2d_6/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
t
conv2d_6/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_6/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_6/Conv2D/concatConcatV2&conv2d_6/Conv2D/strided_slice:output:0(conv2d_6/Conv2D/concat/values_1:output:0$conv2d_6/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:•
conv2d_6/Conv2D/Reshape_1Reshapeconv2d_6/Conv2D/Conv2D:output:0conv2d_6/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
!conv2d_6/squeeze_batch_dims/ShapeShape"conv2d_6/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_6/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_6/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_6/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_6/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_6/squeeze_batch_dims/Shape:output:08conv2d_6/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_6/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_6/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_6/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Ѕ
#conv2d_6/squeeze_batch_dims/ReshapeReshape"conv2d_6/Conv2D/Reshape_1:output:02conv2d_6/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
2conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_6_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0”
#conv2d_6/squeeze_batch_dims/BiasAddBiasAdd,conv2d_6/squeeze_batch_dims/Reshape:output:0:conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АА
+conv2d_6/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_6/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_6/squeeze_batch_dims/concatConcatV22conv2d_6/squeeze_batch_dims/strided_slice:output:04conv2d_6/squeeze_batch_dims/concat/values_1:output:00conv2d_6/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
: 
%conv2d_6/squeeze_batch_dims/Reshape_1Reshape,conv2d_6/squeeze_batch_dims/BiasAdd:output:0+conv2d_6/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АД
conv2d_6/ReluRelu.conv2d_6/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АЪ
	add_3/addAddV2*batch_normalization_3/FusedBatchNormV3:y:0conv2d_6/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аg
activation_3/ReluReluadd_3/add:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€А^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ж
flatten/ReshapeReshapeactivation_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аy
v/MatMul/ReadVariableOpReadVariableOp v_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0}
v/MatMulMatMuldense/BiasAdd:output:0v/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€v
v/BiasAdd/ReadVariableOpReadVariableOp!v_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
	v/BiasAddBiasAddv/MatMul:product:0 v/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T
v/TanhTanhv/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€z
p/MatMul/ReadVariableOpReadVariableOp p_matmul_readvariableop_resource* 
_output_shapes
:
А∞!*
dtype0А
p/MatMulMatMulflatten/Reshape:output:0p/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€∞!w
p/BiasAdd/ReadVariableOpReadVariableOp!p_biasadd_readvariableop_resource*
_output_shapes	
:∞!*
dtype0}
	p/BiasAddBiasAddp/MatMul:product:0 p/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€∞![
	p/SoftmaxSoftmaxp/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€∞!c
IdentityIdentityp/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞![

Identity_1Identity
v/Tanh:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€—
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_2/Conv2D/Conv2D/ReadVariableOp3^conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_3/Conv2D/Conv2D/ReadVariableOp3^conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_4/Conv2D/Conv2D/ReadVariableOp3^conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_5/Conv2D/Conv2D/ReadVariableOp3^conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_6/Conv2D/Conv2D/ReadVariableOp3^conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_7/Conv2D/Conv2D/ReadVariableOp3^conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^p/BiasAdd/ReadVariableOp^p/MatMul/ReadVariableOp^v/BiasAdd/ReadVariableOp^v/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_2/Conv2D/Conv2D/ReadVariableOp%conv2d_2/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_3/Conv2D/Conv2D/ReadVariableOp%conv2d_3/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_4/Conv2D/Conv2D/ReadVariableOp%conv2d_4/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_5/Conv2D/Conv2D/ReadVariableOp%conv2d_5/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_6/Conv2D/Conv2D/ReadVariableOp%conv2d_6/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_7/Conv2D/Conv2D/ReadVariableOp%conv2d_7/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp24
p/BiasAdd/ReadVariableOpp/BiasAdd/ReadVariableOp22
p/MatMul/ReadVariableOpp/MatMul/ReadVariableOp24
v/BiasAdd/ReadVariableOpv/BiasAdd/ReadVariableOp22
v/MatMul/ReadVariableOpv/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
Џ
h
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_3339938

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
»≥
ЯF
#__inference__traced_restore_3341042
file_prefix:
 assignvariableop_conv2d_1_kernel:@.
 assignvariableop_1_conv2d_1_bias:@:
,assignvariableop_2_batch_normalization_gamma:@9
+assignvariableop_3_batch_normalization_beta:@@
2assignvariableop_4_batch_normalization_moving_mean:@D
6assignvariableop_5_batch_normalization_moving_variance:@:
 assignvariableop_6_conv2d_kernel:@,
assignvariableop_7_conv2d_bias:@=
"assignvariableop_8_conv2d_3_kernel:@А/
 assignvariableop_9_conv2d_3_bias:	А>
/assignvariableop_10_batch_normalization_1_gamma:	А=
.assignvariableop_11_batch_normalization_1_beta:	АD
5assignvariableop_12_batch_normalization_1_moving_mean:	АH
9assignvariableop_13_batch_normalization_1_moving_variance:	А>
#assignvariableop_14_conv2d_2_kernel:@А0
!assignvariableop_15_conv2d_2_bias:	А?
#assignvariableop_16_conv2d_5_kernel:АА0
!assignvariableop_17_conv2d_5_bias:	А>
/assignvariableop_18_batch_normalization_2_gamma:	А=
.assignvariableop_19_batch_normalization_2_beta:	АD
5assignvariableop_20_batch_normalization_2_moving_mean:	АH
9assignvariableop_21_batch_normalization_2_moving_variance:	А?
#assignvariableop_22_conv2d_4_kernel:АА0
!assignvariableop_23_conv2d_4_bias:	А?
#assignvariableop_24_conv2d_7_kernel:АА0
!assignvariableop_25_conv2d_7_bias:	А>
/assignvariableop_26_batch_normalization_3_gamma:	А=
.assignvariableop_27_batch_normalization_3_beta:	АD
5assignvariableop_28_batch_normalization_3_moving_mean:	АH
9assignvariableop_29_batch_normalization_3_moving_variance:	А?
#assignvariableop_30_conv2d_6_kernel:АА0
!assignvariableop_31_conv2d_6_bias:	А4
 assignvariableop_32_dense_kernel:
АА-
assignvariableop_33_dense_bias:	А0
assignvariableop_34_p_kernel:
А∞!)
assignvariableop_35_p_bias:	∞!/
assignvariableop_36_v_kernel:	А(
assignvariableop_37_v_bias:'
assignvariableop_38_adam_iter:	 )
assignvariableop_39_adam_beta_1: )
assignvariableop_40_adam_beta_2: (
assignvariableop_41_adam_decay: 0
&assignvariableop_42_adam_learning_rate: #
assignvariableop_43_total: #
assignvariableop_44_count: %
assignvariableop_45_total_1: %
assignvariableop_46_count_1: %
assignvariableop_47_total_2: %
assignvariableop_48_count_2: D
*assignvariableop_49_adam_conv2d_1_kernel_m:@6
(assignvariableop_50_adam_conv2d_1_bias_m:@B
4assignvariableop_51_adam_batch_normalization_gamma_m:@A
3assignvariableop_52_adam_batch_normalization_beta_m:@B
(assignvariableop_53_adam_conv2d_kernel_m:@4
&assignvariableop_54_adam_conv2d_bias_m:@E
*assignvariableop_55_adam_conv2d_3_kernel_m:@А7
(assignvariableop_56_adam_conv2d_3_bias_m:	АE
6assignvariableop_57_adam_batch_normalization_1_gamma_m:	АD
5assignvariableop_58_adam_batch_normalization_1_beta_m:	АE
*assignvariableop_59_adam_conv2d_2_kernel_m:@А7
(assignvariableop_60_adam_conv2d_2_bias_m:	АF
*assignvariableop_61_adam_conv2d_5_kernel_m:АА7
(assignvariableop_62_adam_conv2d_5_bias_m:	АE
6assignvariableop_63_adam_batch_normalization_2_gamma_m:	АD
5assignvariableop_64_adam_batch_normalization_2_beta_m:	АF
*assignvariableop_65_adam_conv2d_4_kernel_m:АА7
(assignvariableop_66_adam_conv2d_4_bias_m:	АF
*assignvariableop_67_adam_conv2d_7_kernel_m:АА7
(assignvariableop_68_adam_conv2d_7_bias_m:	АE
6assignvariableop_69_adam_batch_normalization_3_gamma_m:	АD
5assignvariableop_70_adam_batch_normalization_3_beta_m:	АF
*assignvariableop_71_adam_conv2d_6_kernel_m:АА7
(assignvariableop_72_adam_conv2d_6_bias_m:	А;
'assignvariableop_73_adam_dense_kernel_m:
АА4
%assignvariableop_74_adam_dense_bias_m:	А7
#assignvariableop_75_adam_p_kernel_m:
А∞!0
!assignvariableop_76_adam_p_bias_m:	∞!6
#assignvariableop_77_adam_v_kernel_m:	А/
!assignvariableop_78_adam_v_bias_m:D
*assignvariableop_79_adam_conv2d_1_kernel_v:@6
(assignvariableop_80_adam_conv2d_1_bias_v:@B
4assignvariableop_81_adam_batch_normalization_gamma_v:@A
3assignvariableop_82_adam_batch_normalization_beta_v:@B
(assignvariableop_83_adam_conv2d_kernel_v:@4
&assignvariableop_84_adam_conv2d_bias_v:@E
*assignvariableop_85_adam_conv2d_3_kernel_v:@А7
(assignvariableop_86_adam_conv2d_3_bias_v:	АE
6assignvariableop_87_adam_batch_normalization_1_gamma_v:	АD
5assignvariableop_88_adam_batch_normalization_1_beta_v:	АE
*assignvariableop_89_adam_conv2d_2_kernel_v:@А7
(assignvariableop_90_adam_conv2d_2_bias_v:	АF
*assignvariableop_91_adam_conv2d_5_kernel_v:АА7
(assignvariableop_92_adam_conv2d_5_bias_v:	АE
6assignvariableop_93_adam_batch_normalization_2_gamma_v:	АD
5assignvariableop_94_adam_batch_normalization_2_beta_v:	АF
*assignvariableop_95_adam_conv2d_4_kernel_v:АА7
(assignvariableop_96_adam_conv2d_4_bias_v:	АF
*assignvariableop_97_adam_conv2d_7_kernel_v:АА7
(assignvariableop_98_adam_conv2d_7_bias_v:	АE
6assignvariableop_99_adam_batch_normalization_3_gamma_v:	АE
6assignvariableop_100_adam_batch_normalization_3_beta_v:	АG
+assignvariableop_101_adam_conv2d_6_kernel_v:АА8
)assignvariableop_102_adam_conv2d_6_bias_v:	А<
(assignvariableop_103_adam_dense_kernel_v:
АА5
&assignvariableop_104_adam_dense_bias_v:	А8
$assignvariableop_105_adam_p_kernel_v:
А∞!1
"assignvariableop_106_adam_p_bias_v:	∞!7
$assignvariableop_107_adam_v_kernel_v:	А0
"assignvariableop_108_adam_v_bias_v:
identity_110ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_100ҐAssignVariableOp_101ҐAssignVariableOp_102ҐAssignVariableOp_103ҐAssignVariableOp_104ҐAssignVariableOp_105ҐAssignVariableOp_106ҐAssignVariableOp_107ҐAssignVariableOp_108ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_84ҐAssignVariableOp_85ҐAssignVariableOp_86ҐAssignVariableOp_87ҐAssignVariableOp_88ҐAssignVariableOp_89ҐAssignVariableOp_9ҐAssignVariableOp_90ҐAssignVariableOp_91ҐAssignVariableOp_92ҐAssignVariableOp_93ҐAssignVariableOp_94ҐAssignVariableOp_95ҐAssignVariableOp_96ҐAssignVariableOp_97ҐAssignVariableOp_98ҐAssignVariableOp_99®=
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:n*
dtype0*ќ<
valueƒ<BЅ<nB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHѕ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:n*
dtype0*с
valueзBдnB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B «
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ќ
_output_shapesї
Є::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*|
dtypesr
p2n	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2d_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_1_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_1_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_5_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv2d_5_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_2_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_2_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_2_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_2_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv2d_4_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_23AssignVariableOp!assignvariableop_23_conv2d_4_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_7_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_7_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_3_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_3_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_3_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_3_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_6_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv2d_6_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_32AssignVariableOp assignvariableop_32_dense_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_33AssignVariableOpassignvariableop_33_dense_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_34AssignVariableOpassignvariableop_34_p_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_35AssignVariableOpassignvariableop_35_p_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_36AssignVariableOpassignvariableop_36_v_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_37AssignVariableOpassignvariableop_37_v_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_iterIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_beta_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_beta_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_decayIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_learning_rateIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_43AssignVariableOpassignvariableop_43_totalIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_44AssignVariableOpassignvariableop_44_countIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_1Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_2Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_2Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv2d_1_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv2d_1_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_51AssignVariableOp4assignvariableop_51_adam_batch_normalization_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_batch_normalization_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_conv2d_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_conv2d_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv2d_3_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv2d_3_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_1_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_batch_normalization_1_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv2d_2_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv2d_2_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv2d_5_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_conv2d_5_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_63AssignVariableOp6assignvariableop_63_adam_batch_normalization_2_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_64AssignVariableOp5assignvariableop_64_adam_batch_normalization_2_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_conv2d_4_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_conv2d_4_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv2d_7_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv2d_7_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_3_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_batch_normalization_3_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_conv2d_6_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_conv2d_6_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_73AssignVariableOp'assignvariableop_73_adam_dense_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_74AssignVariableOp%assignvariableop_74_adam_dense_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_75AssignVariableOp#assignvariableop_75_adam_p_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_76AssignVariableOp!assignvariableop_76_adam_p_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_77AssignVariableOp#assignvariableop_77_adam_v_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_78AssignVariableOp!assignvariableop_78_adam_v_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_conv2d_1_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_conv2d_1_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_81AssignVariableOp4assignvariableop_81_adam_batch_normalization_gamma_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_82AssignVariableOp3assignvariableop_82_adam_batch_normalization_beta_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_83AssignVariableOp(assignvariableop_83_adam_conv2d_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_84AssignVariableOp&assignvariableop_84_adam_conv2d_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_conv2d_3_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_conv2d_3_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_87AssignVariableOp6assignvariableop_87_adam_batch_normalization_1_gamma_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_88AssignVariableOp5assignvariableop_88_adam_batch_normalization_1_beta_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_conv2d_2_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_conv2d_2_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_conv2d_5_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_conv2d_5_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_93AssignVariableOp6assignvariableop_93_adam_batch_normalization_2_gamma_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_94AssignVariableOp5assignvariableop_94_adam_batch_normalization_2_beta_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_95AssignVariableOp*assignvariableop_95_adam_conv2d_4_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_96AssignVariableOp(assignvariableop_96_adam_conv2d_4_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_conv2d_7_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_conv2d_7_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_99AssignVariableOp6assignvariableop_99_adam_batch_normalization_3_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_100AssignVariableOp6assignvariableop_100_adam_batch_normalization_3_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_101AssignVariableOp+assignvariableop_101_adam_conv2d_6_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_102AssignVariableOp)assignvariableop_102_adam_conv2d_6_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_103AssignVariableOp(assignvariableop_103_adam_dense_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_104AssignVariableOp&assignvariableop_104_adam_dense_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_105AssignVariableOp$assignvariableop_105_adam_p_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_106AssignVariableOp"assignvariableop_106_adam_p_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_107AssignVariableOp$assignvariableop_107_adam_v_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_108AssignVariableOp"assignvariableop_108_adam_v_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ј
Identity_109Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_110IdentityIdentity_109:output:0^NoOp_1*
T0*
_output_shapes
: £
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_110Identity_110:output:0*с
_input_shapesя
№: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
™
°
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3340024

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0т
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
‘
Ќ	
%__inference_signature_wrapper_3339582
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@
	unknown_6:@$
	unknown_7:@А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А%

unknown_13:@А

unknown_14:	А&

unknown_15:АА

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А&

unknown_29:АА

unknown_30:	А

unknown_31:
АА

unknown_32:	А

unknown_33:	А

unknown_34:

unknown_35:
А∞!

unknown_36:	∞!
identity

identity_1ИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':€€€€€€€€€∞!:€€€€€€€€€*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_3337081p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞!q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:€€€€€€€€€
!
_user_specified_name	input_1
э
j
@__inference_add_layer_call_and_return_conditional_losses_3337473

inputs
inputs_1
identity\
addAddV2inputsinputs_1*
T0*3
_output_shapes!
:€€€€€€€€€@[
IdentityIdentityadd:z:0*
T0*3
_output_shapes!
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€@:€€€€€€€€€@:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs:[W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
л
ќ	
'__inference_model_layer_call_fn_3338845

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@
	unknown_6:@$
	unknown_7:@А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А%

unknown_13:@А

unknown_14:	А&

unknown_15:АА

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А&

unknown_29:АА

unknown_30:	А

unknown_31:
АА

unknown_32:	А

unknown_33:	А

unknown_34:

unknown_35:
А∞!

unknown_36:	∞!
identity

identity_1ИҐStatefulPartitionedCall”
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':€€€€€€€€€∞!:€€€€€€€€€*@
_read_only_resource_inputs"
 	
 !"#$%&*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_3338293p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞!q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
ы
c
G__inference_activation_layer_call_and_return_conditional_losses_3339750

inputs
identityR
ReluReluinputs*
T0*3
_output_shapes!
:€€€€€€€€€@f
IdentityIdentityRelu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
–
`
D__inference_flatten_layer_call_and_return_conditional_losses_3337797

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
¶

т
>__inference_p_layer_call_and_return_conditional_losses_3337843

inputs2
matmul_readvariableop_resource:
А∞!.
biasadd_readvariableop_resource:	∞!
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А∞!*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€∞!s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:∞!*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€∞!W
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€∞!a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞!w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
є
E
)__inference_flatten_layer_call_fn_3340289

inputs
identity∞
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3337797a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
¶

т
>__inference_p_layer_call_and_return_conditional_losses_3340334

inputs2
matmul_readvariableop_resource:
А∞!.
biasadd_readvariableop_resource:	∞!
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А∞!*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€∞!s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:∞!*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€∞!W
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€∞!a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞!w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
і%
µ
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3340262

inputsA
%conv2d_conv2d_readvariableop_resource:ААA
2squeeze_batch_dims_biasadd_readvariableop_resource:	А
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         {
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АМ
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:К
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аa
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ¶
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЩ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Є
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аw
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АС
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
Вz
О
B__inference_model_layer_call_and_return_conditional_losses_3337851

inputs*
conv2d_1_3337414:@
conv2d_1_3337416:@)
batch_normalization_3337419:@)
batch_normalization_3337421:@)
batch_normalization_3337423:@)
batch_normalization_3337425:@(
conv2d_3337462:@
conv2d_3337464:@+
conv2d_3_3337517:@А
conv2d_3_3337519:	А,
batch_normalization_1_3337522:	А,
batch_normalization_1_3337524:	А,
batch_normalization_1_3337526:	А,
batch_normalization_1_3337528:	А+
conv2d_2_3337565:@А
conv2d_2_3337567:	А,
conv2d_5_3337620:АА
conv2d_5_3337622:	А,
batch_normalization_2_3337625:	А,
batch_normalization_2_3337627:	А,
batch_normalization_2_3337629:	А,
batch_normalization_2_3337631:	А,
conv2d_4_3337668:АА
conv2d_4_3337670:	А,
conv2d_7_3337723:АА
conv2d_7_3337725:	А,
batch_normalization_3_3337728:	А,
batch_normalization_3_3337730:	А,
batch_normalization_3_3337732:	А,
batch_normalization_3_3337734:	А,
conv2d_6_3337771:АА
conv2d_6_3337773:	А!
dense_3337810:
АА
dense_3337812:	А
	v_3337827:	А
	v_3337829:
	p_3337844:
А∞!
	p_3337846:	∞!
identity

identity_1ИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐp/StatefulPartitionedCallҐv/StatefulPartitionedCall€
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_3337414conv2d_1_3337416*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3337413М
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_3337419batch_normalization_3337421batch_normalization_3337423batch_normalization_3337425*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3337103ч
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3337462conv2d_3337464*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_3337461У
add/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_3337473я
activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_3337480м
max_pooling3d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_3337154†
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv2d_3_3337517conv2d_3_3337519*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_3337516Щ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_1_3337522batch_normalization_1_3337524batch_normalization_1_3337526batch_normalization_1_3337528*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3337179†
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv2d_2_3337565conv2d_2_3337567*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_3337564Ь
add_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_3337576ж
activation_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_3337583у
max_pooling3d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_3337230Ґ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv2d_5_3337620conv2d_5_3337622*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3337619Щ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_2_3337625batch_normalization_2_3337627batch_normalization_2_3337629batch_normalization_2_3337631*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3337255Ґ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv2d_4_3337668conv2d_4_3337670*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_3337667Ь
add_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_3337679ж
activation_2/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_3337686у
max_pooling3d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_3337306Ґ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv2d_7_3337723conv2d_7_3337725*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3337722Щ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_3_3337728batch_normalization_3_3337730batch_normalization_3_3337732batch_normalization_3_3337734*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3337331Ґ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv2d_6_3337771conv2d_6_3337773*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3337770Ь
add_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_3_layer_call_and_return_conditional_losses_3337782ж
activation_3/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_3337789„
flatten/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3337797В
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_3337810dense_3337812*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3337809ч
v/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0	v_3337827	v_3337829*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_v_layer_call_and_return_conditional_losses_3337826т
p/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	p_3337844	p_3337846*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€∞!*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_p_layer_call_and_return_conditional_losses_3337843r
IdentityIdentity"p/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞!s

Identity_1Identity"v/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€т
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall^p/StatefulPartitionedCall^v/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall26
p/StatefulPartitionedCallp/StatefulPartitionedCall26
v/StatefulPartitionedCallv/StatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
™
°
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3337179

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0т
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ѓ%
і
E__inference_conv2d_3_layer_call_and_return_conditional_losses_3337516

inputs@
%conv2d_conv2d_readvariableop_resource:@АA
2squeeze_batch_dims_biasadd_readvariableop_resource:	А
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Л
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:К
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аa
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   ¶
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЩ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Є
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аw
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АС
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
Е
l
B__inference_add_1_layer_call_and_return_conditional_losses_3337576

inputs
inputs_1
identity]
addAddV2inputsinputs_1*
T0*4
_output_shapes"
 :€€€€€€€€€А\
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
™
°
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3337255

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0т
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
¶
њ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3339686

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
і%
µ
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3337619

inputsA
%conv2d_conv2d_readvariableop_resource:ААA
2squeeze_batch_dims_biasadd_readvariableop_resource:	А
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   {
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АМ
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:К
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аa
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ¶
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЩ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Є
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аw
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АС
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
Н
n
B__inference_add_3_layer_call_and_return_conditional_losses_3340274
inputs_0
inputs_1
identity_
addAddV2inputs_0inputs_1*
T0*4
_output_shapes"
 :€€€€€€€€€А\
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€А:^ Z
4
_output_shapes"
 :€€€€€€€€€А
"
_user_specified_name
inputs/0:^Z
4
_output_shapes"
 :€€€€€€€€€А
"
_user_specified_name
inputs/1
м
S
'__inference_add_2_layer_call_fn_3340090
inputs_0
inputs_1
identity«
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_3337679m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€А:^ Z
4
_output_shapes"
 :€€€€€€€€€А
"
_user_specified_name
inputs/0:^Z
4
_output_shapes"
 :€€€€€€€€€А
"
_user_specified_name
inputs/1
Н
n
B__inference_add_2_layer_call_and_return_conditional_losses_3340096
inputs_0
inputs_1
identity_
addAddV2inputs_0inputs_1*
T0*4
_output_shapes"
 :€€€€€€€€€А\
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€А:^ Z
4
_output_shapes"
 :€€€€€€€€€А
"
_user_specified_name
inputs/0:^Z
4
_output_shapes"
 :€€€€€€€€€А
"
_user_specified_name
inputs/1
в
Q
%__inference_add_layer_call_fn_3339734
inputs_0
inputs_1
identityƒ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_3337473l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€@:€€€€€€€€€@:] Y
3
_output_shapes!
:€€€€€€€€€@
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:€€€€€€€€€@
"
_user_specified_name
inputs/1
Б
e
I__inference_activation_2_layer_call_and_return_conditional_losses_3337686

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :€€€€€€€€€Аg
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
–	
ц
B__inference_dense_layer_call_and_return_conditional_losses_3337809

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ь
Я
*__inference_conv2d_1_layer_call_fn_3339591

inputs!
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3337413{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
ѕ	
÷
7__inference_batch_normalization_2_layer_call_fn_3339993

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3337255Ч
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Н
n
B__inference_add_1_layer_call_and_return_conditional_losses_3339918
inputs_0
inputs_1
identity_
addAddV2inputs_0inputs_1*
T0*4
_output_shapes"
 :€€€€€€€€€А\
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€А:^ Z
4
_output_shapes"
 :€€€€€€€€€А
"
_user_specified_name
inputs/0:^Z
4
_output_shapes"
 :€€€€€€€€€А
"
_user_specified_name
inputs/1
џ
J
.__inference_activation_1_layer_call_fn_3339923

inputs
identityЅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_3337583m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
§%
≤
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3339624

inputs?
%conv2d_conv2d_readvariableop_resource:@@
2squeeze_batch_dims_biasadd_readvariableop_resource:@
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   •
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Ш
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ј
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@q
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€@С
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
ш
Э
(__inference_conv2d_layer_call_fn_3339695

inputs!
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_3337461{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
у
ќ	
'__inference_model_layer_call_fn_3338762

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@
	unknown_6:@$
	unknown_7:@А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А%

unknown_13:@А

unknown_14:	А&

unknown_15:АА

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А&

unknown_29:АА

unknown_30:	А

unknown_31:
АА

unknown_32:	А

unknown_33:	А

unknown_34:

unknown_35:
А∞!

unknown_36:	∞!
identity

identity_1ИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':€€€€€€€€€∞!:€€€€€€€€€*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_3337851p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞!q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
П

р
>__inference_v_layer_call_and_return_conditional_losses_3340354

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Г
Ґ
*__inference_conv2d_7_layer_call_fn_3340125

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3337722|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
Б
e
I__inference_activation_3_layer_call_and_return_conditional_losses_3337789

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :€€€€€€€€€Аg
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
Ў
f
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_3337154

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Г
Ґ
*__inference_conv2d_6_layer_call_fn_3340229

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3337770|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
і%
µ
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3337722

inputsA
%conv2d_conv2d_readvariableop_resource:ААA
2squeeze_batch_dims_biasadd_readvariableop_resource:	А
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         {
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АМ
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:К
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аa
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ¶
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЩ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Є
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аw
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АС
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
ѕ	
÷
7__inference_batch_normalization_3_layer_call_fn_3340171

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3337331Ч
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
і√
Г&
"__inference__wrapped_model_3337081
input_1N
4model_conv2d_1_conv2d_conv2d_readvariableop_resource:@O
Amodel_conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:@?
1model_batch_normalization_readvariableop_resource:@A
3model_batch_normalization_readvariableop_1_resource:@P
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource:@R
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@L
2model_conv2d_conv2d_conv2d_readvariableop_resource:@M
?model_conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:@O
4model_conv2d_3_conv2d_conv2d_readvariableop_resource:@АP
Amodel_conv2d_3_squeeze_batch_dims_biasadd_readvariableop_resource:	АB
3model_batch_normalization_1_readvariableop_resource:	АD
5model_batch_normalization_1_readvariableop_1_resource:	АS
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	АU
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	АO
4model_conv2d_2_conv2d_conv2d_readvariableop_resource:@АP
Amodel_conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource:	АP
4model_conv2d_5_conv2d_conv2d_readvariableop_resource:ААP
Amodel_conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:	АB
3model_batch_normalization_2_readvariableop_resource:	АD
5model_batch_normalization_2_readvariableop_1_resource:	АS
Dmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	АU
Fmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	АP
4model_conv2d_4_conv2d_conv2d_readvariableop_resource:ААP
Amodel_conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:	АP
4model_conv2d_7_conv2d_conv2d_readvariableop_resource:ААP
Amodel_conv2d_7_squeeze_batch_dims_biasadd_readvariableop_resource:	АB
3model_batch_normalization_3_readvariableop_resource:	АD
5model_batch_normalization_3_readvariableop_1_resource:	АS
Dmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АU
Fmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	АP
4model_conv2d_6_conv2d_conv2d_readvariableop_resource:ААP
Amodel_conv2d_6_squeeze_batch_dims_biasadd_readvariableop_resource:	А>
*model_dense_matmul_readvariableop_resource:
АА:
+model_dense_biasadd_readvariableop_resource:	А9
&model_v_matmul_readvariableop_resource:	А5
'model_v_biasadd_readvariableop_resource::
&model_p_matmul_readvariableop_resource:
А∞!6
'model_p_biasadd_readvariableop_resource:	∞!
identity

identity_1ИҐ9model/batch_normalization/FusedBatchNormV3/ReadVariableOpҐ;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ(model/batch_normalization/ReadVariableOpҐ*model/batch_normalization/ReadVariableOp_1Ґ;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ*model/batch_normalization_1/ReadVariableOpҐ,model/batch_normalization_1/ReadVariableOp_1Ґ;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ*model/batch_normalization_2/ReadVariableOpҐ,model/batch_normalization_2/ReadVariableOp_1Ґ;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ*model/batch_normalization_3/ReadVariableOpҐ,model/batch_normalization_3/ReadVariableOp_1Ґ)model/conv2d/Conv2D/Conv2D/ReadVariableOpҐ6model/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ+model/conv2d_1/Conv2D/Conv2D/ReadVariableOpҐ8model/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ+model/conv2d_2/Conv2D/Conv2D/ReadVariableOpҐ8model/conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ+model/conv2d_3/Conv2D/Conv2D/ReadVariableOpҐ8model/conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ+model/conv2d_4/Conv2D/Conv2D/ReadVariableOpҐ8model/conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ+model/conv2d_5/Conv2D/Conv2D/ReadVariableOpҐ8model/conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ+model/conv2d_6/Conv2D/Conv2D/ReadVariableOpҐ8model/conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ+model/conv2d_7/Conv2D/Conv2D/ReadVariableOpҐ8model/conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ"model/dense/BiasAdd/ReadVariableOpҐ!model/dense/MatMul/ReadVariableOpҐmodel/p/BiasAdd/ReadVariableOpҐmodel/p/MatMul/ReadVariableOpҐmodel/v/BiasAdd/ReadVariableOpҐmodel/v/MatMul/ReadVariableOpR
model/conv2d_1/Conv2D/ShapeShapeinput_1*
T0*
_output_shapes
:s
)model/conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
+model/conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€u
+model/conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
#model/conv2d_1/Conv2D/strided_sliceStridedSlice$model/conv2d_1/Conv2D/Shape:output:02model/conv2d_1/Conv2D/strided_slice/stack:output:04model/conv2d_1/Conv2D/strided_slice/stack_1:output:04model/conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
#model/conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Щ
model/conv2d_1/Conv2D/ReshapeReshapeinput_1,model/conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€®
+model/conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp4model_conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0е
model/conv2d_1/Conv2D/Conv2DConv2D&model/conv2d_1/Conv2D/Reshape:output:03model/conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
z
%model/conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   l
!model/conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€а
model/conv2d_1/Conv2D/concatConcatV2,model/conv2d_1/Conv2D/strided_slice:output:0.model/conv2d_1/Conv2D/concat/values_1:output:0*model/conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:ґ
model/conv2d_1/Conv2D/Reshape_1Reshape%model/conv2d_1/Conv2D/Conv2D:output:0%model/conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@
'model/conv2d_1/squeeze_batch_dims/ShapeShape(model/conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:
5model/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
7model/conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€Б
7model/conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
/model/conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice0model/conv2d_1/squeeze_batch_dims/Shape:output:0>model/conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0@model/conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0@model/conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskИ
/model/conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   “
)model/conv2d_1/squeeze_batch_dims/ReshapeReshape(model/conv2d_1/Conv2D/Reshape_1:output:08model/conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@ґ
8model/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpAmodel_conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0д
)model/conv2d_1/squeeze_batch_dims/BiasAddBiasAdd2model/conv2d_1/squeeze_batch_dims/Reshape:output:0@model/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@Ж
1model/conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   x
-model/conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Р
(model/conv2d_1/squeeze_batch_dims/concatConcatV28model/conv2d_1/squeeze_batch_dims/strided_slice:output:0:model/conv2d_1/squeeze_batch_dims/concat/values_1:output:06model/conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:џ
+model/conv2d_1/squeeze_batch_dims/Reshape_1Reshape2model/conv2d_1/squeeze_batch_dims/BiasAdd:output:01model/conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@П
model/conv2d_1/ReluRelu4model/conv2d_1/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@Ц
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0Ъ
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0Є
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0п
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3!model/conv2d_1/Relu:activations:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
is_training( P
model/conv2d/Conv2D/ShapeShapeinput_1*
T0*
_output_shapes
:q
'model/conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
)model/conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€s
)model/conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≥
!model/conv2d/Conv2D/strided_sliceStridedSlice"model/conv2d/Conv2D/Shape:output:00model/conv2d/Conv2D/strided_slice/stack:output:02model/conv2d/Conv2D/strided_slice/stack_1:output:02model/conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskz
!model/conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Х
model/conv2d/Conv2D/ReshapeReshapeinput_1*model/conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€§
)model/conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp2model_conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0я
model/conv2d/Conv2D/Conv2DConv2D$model/conv2d/Conv2D/Reshape:output:01model/conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
x
#model/conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   j
model/conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ў
model/conv2d/Conv2D/concatConcatV2*model/conv2d/Conv2D/strided_slice:output:0,model/conv2d/Conv2D/concat/values_1:output:0(model/conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:∞
model/conv2d/Conv2D/Reshape_1Reshape#model/conv2d/Conv2D/Conv2D:output:0#model/conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@{
%model/conv2d/squeeze_batch_dims/ShapeShape&model/conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:}
3model/conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: И
5model/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€
5model/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
-model/conv2d/squeeze_batch_dims/strided_sliceStridedSlice.model/conv2d/squeeze_batch_dims/Shape:output:0<model/conv2d/squeeze_batch_dims/strided_slice/stack:output:0>model/conv2d/squeeze_batch_dims/strided_slice/stack_1:output:0>model/conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskЖ
-model/conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   ћ
'model/conv2d/squeeze_batch_dims/ReshapeReshape&model/conv2d/Conv2D/Reshape_1:output:06model/conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@≤
6model/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp?model_conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ё
'model/conv2d/squeeze_batch_dims/BiasAddBiasAdd0model/conv2d/squeeze_batch_dims/Reshape:output:0>model/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@Д
/model/conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   v
+model/conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€И
&model/conv2d/squeeze_batch_dims/concatConcatV26model/conv2d/squeeze_batch_dims/strided_slice:output:08model/conv2d/squeeze_batch_dims/concat/values_1:output:04model/conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:’
)model/conv2d/squeeze_batch_dims/Reshape_1Reshape0model/conv2d/squeeze_batch_dims/BiasAdd:output:0/model/conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@Л
model/conv2d/ReluRelu2model/conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@•
model/add/addAddV2.model/batch_normalization/FusedBatchNormV3:y:0model/conv2d/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€@n
model/activation/ReluRelumodel/add/add:z:0*
T0*3
_output_shapes!
:€€€€€€€€€@Ћ
model/max_pooling3d/MaxPool3D	MaxPool3D#model/activation/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€@*
ksize	
*
paddingVALID*
strides	
q
model/conv2d_3/Conv2D/ShapeShape&model/max_pooling3d/MaxPool3D:output:0*
T0*
_output_shapes
:s
)model/conv2d_3/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
+model/conv2d_3/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€u
+model/conv2d_3/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
#model/conv2d_3/Conv2D/strided_sliceStridedSlice$model/conv2d_3/Conv2D/Shape:output:02model/conv2d_3/Conv2D/strided_slice/stack:output:04model/conv2d_3/Conv2D/strided_slice/stack_1:output:04model/conv2d_3/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
#model/conv2d_3/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   Є
model/conv2d_3/Conv2D/ReshapeReshape&model/max_pooling3d/MaxPool3D:output:0,model/conv2d_3/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@©
+model/conv2d_3/Conv2D/Conv2D/ReadVariableOpReadVariableOp4model_conv2d_3_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0ж
model/conv2d_3/Conv2D/Conv2DConv2D&model/conv2d_3/Conv2D/Reshape:output:03model/conv2d_3/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
z
%model/conv2d_3/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   l
!model/conv2d_3/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€а
model/conv2d_3/Conv2D/concatConcatV2,model/conv2d_3/Conv2D/strided_slice:output:0.model/conv2d_3/Conv2D/concat/values_1:output:0*model/conv2d_3/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Ј
model/conv2d_3/Conv2D/Reshape_1Reshape%model/conv2d_3/Conv2D/Conv2D:output:0%model/conv2d_3/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А
'model/conv2d_3/squeeze_batch_dims/ShapeShape(model/conv2d_3/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:
5model/conv2d_3/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
7model/conv2d_3/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€Б
7model/conv2d_3/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
/model/conv2d_3/squeeze_batch_dims/strided_sliceStridedSlice0model/conv2d_3/squeeze_batch_dims/Shape:output:0>model/conv2d_3/squeeze_batch_dims/strided_slice/stack:output:0@model/conv2d_3/squeeze_batch_dims/strided_slice/stack_1:output:0@model/conv2d_3/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskИ
/model/conv2d_3/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   ”
)model/conv2d_3/squeeze_batch_dims/ReshapeReshape(model/conv2d_3/Conv2D/Reshape_1:output:08model/conv2d_3/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЈ
8model/conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpAmodel_conv2d_3_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
)model/conv2d_3/squeeze_batch_dims/BiasAddBiasAdd2model/conv2d_3/squeeze_batch_dims/Reshape:output:0@model/conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЖ
1model/conv2d_3/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   x
-model/conv2d_3/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Р
(model/conv2d_3/squeeze_batch_dims/concatConcatV28model/conv2d_3/squeeze_batch_dims/strided_slice:output:0:model/conv2d_3/squeeze_batch_dims/concat/values_1:output:06model/conv2d_3/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:№
+model/conv2d_3/squeeze_batch_dims/Reshape_1Reshape2model/conv2d_3/squeeze_batch_dims/BiasAdd:output:01model/conv2d_3/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АР
model/conv2d_3/ReluRelu4model/conv2d_3/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АЫ
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Я
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:А*
dtype0љ
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Ѕ
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ю
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!model/conv2d_3/Relu:activations:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( q
model/conv2d_2/Conv2D/ShapeShape&model/max_pooling3d/MaxPool3D:output:0*
T0*
_output_shapes
:s
)model/conv2d_2/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
+model/conv2d_2/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€u
+model/conv2d_2/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
#model/conv2d_2/Conv2D/strided_sliceStridedSlice$model/conv2d_2/Conv2D/Shape:output:02model/conv2d_2/Conv2D/strided_slice/stack:output:04model/conv2d_2/Conv2D/strided_slice/stack_1:output:04model/conv2d_2/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
#model/conv2d_2/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   Є
model/conv2d_2/Conv2D/ReshapeReshape&model/max_pooling3d/MaxPool3D:output:0,model/conv2d_2/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@©
+model/conv2d_2/Conv2D/Conv2D/ReadVariableOpReadVariableOp4model_conv2d_2_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0ж
model/conv2d_2/Conv2D/Conv2DConv2D&model/conv2d_2/Conv2D/Reshape:output:03model/conv2d_2/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
z
%model/conv2d_2/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   l
!model/conv2d_2/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€а
model/conv2d_2/Conv2D/concatConcatV2,model/conv2d_2/Conv2D/strided_slice:output:0.model/conv2d_2/Conv2D/concat/values_1:output:0*model/conv2d_2/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Ј
model/conv2d_2/Conv2D/Reshape_1Reshape%model/conv2d_2/Conv2D/Conv2D:output:0%model/conv2d_2/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А
'model/conv2d_2/squeeze_batch_dims/ShapeShape(model/conv2d_2/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:
5model/conv2d_2/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
7model/conv2d_2/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€Б
7model/conv2d_2/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
/model/conv2d_2/squeeze_batch_dims/strided_sliceStridedSlice0model/conv2d_2/squeeze_batch_dims/Shape:output:0>model/conv2d_2/squeeze_batch_dims/strided_slice/stack:output:0@model/conv2d_2/squeeze_batch_dims/strided_slice/stack_1:output:0@model/conv2d_2/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskИ
/model/conv2d_2/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   ”
)model/conv2d_2/squeeze_batch_dims/ReshapeReshape(model/conv2d_2/Conv2D/Reshape_1:output:08model/conv2d_2/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЈ
8model/conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpAmodel_conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
)model/conv2d_2/squeeze_batch_dims/BiasAddBiasAdd2model/conv2d_2/squeeze_batch_dims/Reshape:output:0@model/conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЖ
1model/conv2d_2/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   x
-model/conv2d_2/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Р
(model/conv2d_2/squeeze_batch_dims/concatConcatV28model/conv2d_2/squeeze_batch_dims/strided_slice:output:0:model/conv2d_2/squeeze_batch_dims/concat/values_1:output:06model/conv2d_2/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:№
+model/conv2d_2/squeeze_batch_dims/Reshape_1Reshape2model/conv2d_2/squeeze_batch_dims/BiasAdd:output:01model/conv2d_2/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АР
model/conv2d_2/ReluRelu4model/conv2d_2/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Ађ
model/add_1/addAddV20model/batch_normalization_1/FusedBatchNormV3:y:0!model/conv2d_2/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
model/activation_1/ReluRelumodel/add_1/add:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€А–
model/max_pooling3d_1/MaxPool3D	MaxPool3D%model/activation_1/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
ksize	
*
paddingVALID*
strides	
s
model/conv2d_5/Conv2D/ShapeShape(model/max_pooling3d_1/MaxPool3D:output:0*
T0*
_output_shapes
:s
)model/conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
+model/conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€u
+model/conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
#model/conv2d_5/Conv2D/strided_sliceStridedSlice$model/conv2d_5/Conv2D/Shape:output:02model/conv2d_5/Conv2D/strided_slice/stack:output:04model/conv2d_5/Conv2D/strided_slice/stack_1:output:04model/conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
#model/conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   ї
model/conv2d_5/Conv2D/ReshapeReshape(model/max_pooling3d_1/MaxPool3D:output:0,model/conv2d_5/Conv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А™
+model/conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp4model_conv2d_5_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0ж
model/conv2d_5/Conv2D/Conv2DConv2D&model/conv2d_5/Conv2D/Reshape:output:03model/conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
z
%model/conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         l
!model/conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€а
model/conv2d_5/Conv2D/concatConcatV2,model/conv2d_5/Conv2D/strided_slice:output:0.model/conv2d_5/Conv2D/concat/values_1:output:0*model/conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Ј
model/conv2d_5/Conv2D/Reshape_1Reshape%model/conv2d_5/Conv2D/Conv2D:output:0%model/conv2d_5/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А
'model/conv2d_5/squeeze_batch_dims/ShapeShape(model/conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:
5model/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
7model/conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€Б
7model/conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
/model/conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice0model/conv2d_5/squeeze_batch_dims/Shape:output:0>model/conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0@model/conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0@model/conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskИ
/model/conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ”
)model/conv2d_5/squeeze_batch_dims/ReshapeReshape(model/conv2d_5/Conv2D/Reshape_1:output:08model/conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЈ
8model/conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpAmodel_conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
)model/conv2d_5/squeeze_batch_dims/BiasAddBiasAdd2model/conv2d_5/squeeze_batch_dims/Reshape:output:0@model/conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЖ
1model/conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         x
-model/conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Р
(model/conv2d_5/squeeze_batch_dims/concatConcatV28model/conv2d_5/squeeze_batch_dims/strided_slice:output:0:model/conv2d_5/squeeze_batch_dims/concat/values_1:output:06model/conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:№
+model/conv2d_5/squeeze_batch_dims/Reshape_1Reshape2model/conv2d_5/squeeze_batch_dims/BiasAdd:output:01model/conv2d_5/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АР
model/conv2d_5/ReluRelu4model/conv2d_5/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АЫ
*model/batch_normalization_2/ReadVariableOpReadVariableOp3model_batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype0Я
,model/batch_normalization_2/ReadVariableOp_1ReadVariableOp5model_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype0љ
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Ѕ
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ю
,model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3!model/conv2d_5/Relu:activations:02model/batch_normalization_2/ReadVariableOp:value:04model/batch_normalization_2/ReadVariableOp_1:value:0Cmodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( s
model/conv2d_4/Conv2D/ShapeShape(model/max_pooling3d_1/MaxPool3D:output:0*
T0*
_output_shapes
:s
)model/conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
+model/conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€u
+model/conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
#model/conv2d_4/Conv2D/strided_sliceStridedSlice$model/conv2d_4/Conv2D/Shape:output:02model/conv2d_4/Conv2D/strided_slice/stack:output:04model/conv2d_4/Conv2D/strided_slice/stack_1:output:04model/conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
#model/conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   ї
model/conv2d_4/Conv2D/ReshapeReshape(model/max_pooling3d_1/MaxPool3D:output:0,model/conv2d_4/Conv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А™
+model/conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp4model_conv2d_4_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0ж
model/conv2d_4/Conv2D/Conv2DConv2D&model/conv2d_4/Conv2D/Reshape:output:03model/conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
z
%model/conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         l
!model/conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€а
model/conv2d_4/Conv2D/concatConcatV2,model/conv2d_4/Conv2D/strided_slice:output:0.model/conv2d_4/Conv2D/concat/values_1:output:0*model/conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Ј
model/conv2d_4/Conv2D/Reshape_1Reshape%model/conv2d_4/Conv2D/Conv2D:output:0%model/conv2d_4/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А
'model/conv2d_4/squeeze_batch_dims/ShapeShape(model/conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:
5model/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
7model/conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€Б
7model/conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
/model/conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice0model/conv2d_4/squeeze_batch_dims/Shape:output:0>model/conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0@model/conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0@model/conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskИ
/model/conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ”
)model/conv2d_4/squeeze_batch_dims/ReshapeReshape(model/conv2d_4/Conv2D/Reshape_1:output:08model/conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЈ
8model/conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpAmodel_conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
)model/conv2d_4/squeeze_batch_dims/BiasAddBiasAdd2model/conv2d_4/squeeze_batch_dims/Reshape:output:0@model/conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЖ
1model/conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         x
-model/conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Р
(model/conv2d_4/squeeze_batch_dims/concatConcatV28model/conv2d_4/squeeze_batch_dims/strided_slice:output:0:model/conv2d_4/squeeze_batch_dims/concat/values_1:output:06model/conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:№
+model/conv2d_4/squeeze_batch_dims/Reshape_1Reshape2model/conv2d_4/squeeze_batch_dims/BiasAdd:output:01model/conv2d_4/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АР
model/conv2d_4/ReluRelu4model/conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Ађ
model/add_2/addAddV20model/batch_normalization_2/FusedBatchNormV3:y:0!model/conv2d_4/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
model/activation_2/ReluRelumodel/add_2/add:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€А–
model/max_pooling3d_2/MaxPool3D	MaxPool3D%model/activation_2/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
ksize	
*
paddingVALID*
strides	
s
model/conv2d_7/Conv2D/ShapeShape(model/max_pooling3d_2/MaxPool3D:output:0*
T0*
_output_shapes
:s
)model/conv2d_7/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
+model/conv2d_7/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€u
+model/conv2d_7/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
#model/conv2d_7/Conv2D/strided_sliceStridedSlice$model/conv2d_7/Conv2D/Shape:output:02model/conv2d_7/Conv2D/strided_slice/stack:output:04model/conv2d_7/Conv2D/strided_slice/stack_1:output:04model/conv2d_7/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
#model/conv2d_7/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ї
model/conv2d_7/Conv2D/ReshapeReshape(model/max_pooling3d_2/MaxPool3D:output:0,model/conv2d_7/Conv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А™
+model/conv2d_7/Conv2D/Conv2D/ReadVariableOpReadVariableOp4model_conv2d_7_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0ж
model/conv2d_7/Conv2D/Conv2DConv2D&model/conv2d_7/Conv2D/Reshape:output:03model/conv2d_7/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
z
%model/conv2d_7/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         l
!model/conv2d_7/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€а
model/conv2d_7/Conv2D/concatConcatV2,model/conv2d_7/Conv2D/strided_slice:output:0.model/conv2d_7/Conv2D/concat/values_1:output:0*model/conv2d_7/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Ј
model/conv2d_7/Conv2D/Reshape_1Reshape%model/conv2d_7/Conv2D/Conv2D:output:0%model/conv2d_7/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А
'model/conv2d_7/squeeze_batch_dims/ShapeShape(model/conv2d_7/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:
5model/conv2d_7/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
7model/conv2d_7/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€Б
7model/conv2d_7/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
/model/conv2d_7/squeeze_batch_dims/strided_sliceStridedSlice0model/conv2d_7/squeeze_batch_dims/Shape:output:0>model/conv2d_7/squeeze_batch_dims/strided_slice/stack:output:0@model/conv2d_7/squeeze_batch_dims/strided_slice/stack_1:output:0@model/conv2d_7/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskИ
/model/conv2d_7/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ”
)model/conv2d_7/squeeze_batch_dims/ReshapeReshape(model/conv2d_7/Conv2D/Reshape_1:output:08model/conv2d_7/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЈ
8model/conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpAmodel_conv2d_7_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
)model/conv2d_7/squeeze_batch_dims/BiasAddBiasAdd2model/conv2d_7/squeeze_batch_dims/Reshape:output:0@model/conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЖ
1model/conv2d_7/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         x
-model/conv2d_7/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Р
(model/conv2d_7/squeeze_batch_dims/concatConcatV28model/conv2d_7/squeeze_batch_dims/strided_slice:output:0:model/conv2d_7/squeeze_batch_dims/concat/values_1:output:06model/conv2d_7/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:№
+model/conv2d_7/squeeze_batch_dims/Reshape_1Reshape2model/conv2d_7/squeeze_batch_dims/BiasAdd:output:01model/conv2d_7/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АР
model/conv2d_7/ReluRelu4model/conv2d_7/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АЫ
*model/batch_normalization_3/ReadVariableOpReadVariableOp3model_batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0Я
,model/batch_normalization_3/ReadVariableOp_1ReadVariableOp5model_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0љ
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Ѕ
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ю
,model/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3!model/conv2d_7/Relu:activations:02model/batch_normalization_3/ReadVariableOp:value:04model/batch_normalization_3/ReadVariableOp_1:value:0Cmodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( s
model/conv2d_6/Conv2D/ShapeShape(model/max_pooling3d_2/MaxPool3D:output:0*
T0*
_output_shapes
:s
)model/conv2d_6/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
+model/conv2d_6/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€u
+model/conv2d_6/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
#model/conv2d_6/Conv2D/strided_sliceStridedSlice$model/conv2d_6/Conv2D/Shape:output:02model/conv2d_6/Conv2D/strided_slice/stack:output:04model/conv2d_6/Conv2D/strided_slice/stack_1:output:04model/conv2d_6/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
#model/conv2d_6/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ї
model/conv2d_6/Conv2D/ReshapeReshape(model/max_pooling3d_2/MaxPool3D:output:0,model/conv2d_6/Conv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А™
+model/conv2d_6/Conv2D/Conv2D/ReadVariableOpReadVariableOp4model_conv2d_6_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0ж
model/conv2d_6/Conv2D/Conv2DConv2D&model/conv2d_6/Conv2D/Reshape:output:03model/conv2d_6/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
z
%model/conv2d_6/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         l
!model/conv2d_6/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€а
model/conv2d_6/Conv2D/concatConcatV2,model/conv2d_6/Conv2D/strided_slice:output:0.model/conv2d_6/Conv2D/concat/values_1:output:0*model/conv2d_6/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Ј
model/conv2d_6/Conv2D/Reshape_1Reshape%model/conv2d_6/Conv2D/Conv2D:output:0%model/conv2d_6/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А
'model/conv2d_6/squeeze_batch_dims/ShapeShape(model/conv2d_6/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:
5model/conv2d_6/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
7model/conv2d_6/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€Б
7model/conv2d_6/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
/model/conv2d_6/squeeze_batch_dims/strided_sliceStridedSlice0model/conv2d_6/squeeze_batch_dims/Shape:output:0>model/conv2d_6/squeeze_batch_dims/strided_slice/stack:output:0@model/conv2d_6/squeeze_batch_dims/strided_slice/stack_1:output:0@model/conv2d_6/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskИ
/model/conv2d_6/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ”
)model/conv2d_6/squeeze_batch_dims/ReshapeReshape(model/conv2d_6/Conv2D/Reshape_1:output:08model/conv2d_6/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЈ
8model/conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpAmodel_conv2d_6_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
)model/conv2d_6/squeeze_batch_dims/BiasAddBiasAdd2model/conv2d_6/squeeze_batch_dims/Reshape:output:0@model/conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АЖ
1model/conv2d_6/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         x
-model/conv2d_6/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Р
(model/conv2d_6/squeeze_batch_dims/concatConcatV28model/conv2d_6/squeeze_batch_dims/strided_slice:output:0:model/conv2d_6/squeeze_batch_dims/concat/values_1:output:06model/conv2d_6/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:№
+model/conv2d_6/squeeze_batch_dims/Reshape_1Reshape2model/conv2d_6/squeeze_batch_dims/BiasAdd:output:01model/conv2d_6/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АР
model/conv2d_6/ReluRelu4model/conv2d_6/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Ађ
model/add_3/addAddV20model/batch_normalization_3/FusedBatchNormV3:y:0!model/conv2d_6/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
model/activation_3/ReluRelumodel/add_3/add:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аd
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ш
model/flatten/ReshapeReshape%model/activation_3/Relu:activations:0model/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АО
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ъ
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЛ
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ы
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЕ
model/v/MatMul/ReadVariableOpReadVariableOp&model_v_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0П
model/v/MatMulMatMulmodel/dense/BiasAdd:output:0%model/v/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
model/v/BiasAdd/ReadVariableOpReadVariableOp'model_v_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
model/v/BiasAddBiasAddmodel/v/MatMul:product:0&model/v/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`
model/v/TanhTanhmodel/v/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
model/p/MatMul/ReadVariableOpReadVariableOp&model_p_matmul_readvariableop_resource* 
_output_shapes
:
А∞!*
dtype0Т
model/p/MatMulMatMulmodel/flatten/Reshape:output:0%model/p/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€∞!Г
model/p/BiasAdd/ReadVariableOpReadVariableOp'model_p_biasadd_readvariableop_resource*
_output_shapes	
:∞!*
dtype0П
model/p/BiasAddBiasAddmodel/p/MatMul:product:0&model/p/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€∞!g
model/p/SoftmaxSoftmaxmodel/p/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€∞!i
IdentityIdentitymodel/p/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞!a

Identity_1Identitymodel/v/Tanh:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€µ
NoOpNoOp:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1<^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_2/ReadVariableOp-^model/batch_normalization_2/ReadVariableOp_1<^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_3/ReadVariableOp-^model/batch_normalization_3/ReadVariableOp_1*^model/conv2d/Conv2D/Conv2D/ReadVariableOp7^model/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp,^model/conv2d_1/Conv2D/Conv2D/ReadVariableOp9^model/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp,^model/conv2d_2/Conv2D/Conv2D/ReadVariableOp9^model/conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp,^model/conv2d_3/Conv2D/Conv2D/ReadVariableOp9^model/conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp,^model/conv2d_4/Conv2D/Conv2D/ReadVariableOp9^model/conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp,^model/conv2d_5/Conv2D/Conv2D/ReadVariableOp9^model/conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp,^model/conv2d_6/Conv2D/Conv2D/ReadVariableOp9^model/conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOp,^model/conv2d_7/Conv2D/Conv2D/ReadVariableOp9^model/conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp^model/p/BiasAdd/ReadVariableOp^model/p/MatMul/ReadVariableOp^model/v/BiasAdd/ReadVariableOp^model/v/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12z
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_1/ReadVariableOp*model/batch_normalization_1/ReadVariableOp2\
,model/batch_normalization_1/ReadVariableOp_1,model/batch_normalization_1/ReadVariableOp_12z
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_2/ReadVariableOp*model/batch_normalization_2/ReadVariableOp2\
,model/batch_normalization_2/ReadVariableOp_1,model/batch_normalization_2/ReadVariableOp_12z
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_3/ReadVariableOp*model/batch_normalization_3/ReadVariableOp2\
,model/batch_normalization_3/ReadVariableOp_1,model/batch_normalization_3/ReadVariableOp_12V
)model/conv2d/Conv2D/Conv2D/ReadVariableOp)model/conv2d/Conv2D/Conv2D/ReadVariableOp2p
6model/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp6model/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2Z
+model/conv2d_1/Conv2D/Conv2D/ReadVariableOp+model/conv2d_1/Conv2D/Conv2D/ReadVariableOp2t
8model/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp8model/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2Z
+model/conv2d_2/Conv2D/Conv2D/ReadVariableOp+model/conv2d_2/Conv2D/Conv2D/ReadVariableOp2t
8model/conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp8model/conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp2Z
+model/conv2d_3/Conv2D/Conv2D/ReadVariableOp+model/conv2d_3/Conv2D/Conv2D/ReadVariableOp2t
8model/conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp8model/conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2Z
+model/conv2d_4/Conv2D/Conv2D/ReadVariableOp+model/conv2d_4/Conv2D/Conv2D/ReadVariableOp2t
8model/conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp8model/conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2Z
+model/conv2d_5/Conv2D/Conv2D/ReadVariableOp+model/conv2d_5/Conv2D/Conv2D/ReadVariableOp2t
8model/conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp8model/conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2Z
+model/conv2d_6/Conv2D/Conv2D/ReadVariableOp+model/conv2d_6/Conv2D/Conv2D/ReadVariableOp2t
8model/conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOp8model/conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOp2Z
+model/conv2d_7/Conv2D/Conv2D/ReadVariableOp+model/conv2d_7/Conv2D/Conv2D/ReadVariableOp2t
8model/conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOp8model/conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2@
model/p/BiasAdd/ReadVariableOpmodel/p/BiasAdd/ReadVariableOp2>
model/p/MatMul/ReadVariableOpmodel/p/MatMul/ReadVariableOp2@
model/v/BiasAdd/ReadVariableOpmodel/v/BiasAdd/ReadVariableOp2>
model/v/MatMul/ReadVariableOpmodel/v/MatMul/ReadVariableOp:\ X
3
_output_shapes!
:€€€€€€€€€
!
_user_specified_name	input_1
Џ
h
L__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_3340116

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
љЇ
Ц%
B__inference_model_layer_call_and_return_conditional_losses_3339497

inputsH
.conv2d_1_conv2d_conv2d_readvariableop_resource:@I
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@F
,conv2d_conv2d_conv2d_readvariableop_resource:@G
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource:@I
.conv2d_3_conv2d_conv2d_readvariableop_resource:@АJ
;conv2d_3_squeeze_batch_dims_biasadd_readvariableop_resource:	А<
-batch_normalization_1_readvariableop_resource:	А>
/batch_normalization_1_readvariableop_1_resource:	АM
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	АI
.conv2d_2_conv2d_conv2d_readvariableop_resource:@АJ
;conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource:	АJ
.conv2d_5_conv2d_conv2d_readvariableop_resource:ААJ
;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource:	А<
-batch_normalization_2_readvariableop_resource:	А>
/batch_normalization_2_readvariableop_1_resource:	АM
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	АJ
.conv2d_4_conv2d_conv2d_readvariableop_resource:ААJ
;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource:	АJ
.conv2d_7_conv2d_conv2d_readvariableop_resource:ААJ
;conv2d_7_squeeze_batch_dims_biasadd_readvariableop_resource:	А<
-batch_normalization_3_readvariableop_resource:	А>
/batch_normalization_3_readvariableop_1_resource:	АM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	АJ
.conv2d_6_conv2d_conv2d_readvariableop_resource:ААJ
;conv2d_6_squeeze_batch_dims_biasadd_readvariableop_resource:	А8
$dense_matmul_readvariableop_resource:
АА4
%dense_biasadd_readvariableop_resource:	А3
 v_matmul_readvariableop_resource:	А/
!v_biasadd_readvariableop_resource:4
 p_matmul_readvariableop_resource:
А∞!0
!p_biasadd_readvariableop_resource:	∞!
identity

identity_1ИҐ"batch_normalization/AssignNewValueҐ$batch_normalization/AssignNewValue_1Ґ3batch_normalization/FusedBatchNormV3/ReadVariableOpҐ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ"batch_normalization/ReadVariableOpҐ$batch_normalization/ReadVariableOp_1Ґ$batch_normalization_1/AssignNewValueҐ&batch_normalization_1/AssignNewValue_1Ґ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_1/ReadVariableOpҐ&batch_normalization_1/ReadVariableOp_1Ґ$batch_normalization_2/AssignNewValueҐ&batch_normalization_2/AssignNewValue_1Ґ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґ$batch_normalization_3/AssignNewValueҐ&batch_normalization_3/AssignNewValue_1Ґ5batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_3/ReadVariableOpҐ&batch_normalization_3/ReadVariableOp_1Ґ#conv2d/Conv2D/Conv2D/ReadVariableOpҐ0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_1/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_2/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_3/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_4/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_5/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_6/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOpҐ%conv2d_7/Conv2D/Conv2D/ReadVariableOpҐ2conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐp/BiasAdd/ReadVariableOpҐp/MatMul/ReadVariableOpҐv/BiasAdd/ReadVariableOpҐv/MatMul/ReadVariableOpK
conv2d_1/Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:m
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         М
conv2d_1/Conv2D/ReshapeReshapeinputs&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€Ь
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0”
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
t
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   f
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:§
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@s
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   ј
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@™
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0“
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@А
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   r
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:…
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@Г
conv2d_1/ReluRelu.conv2d_1/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@К
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0О
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0ђ
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0∞
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ў
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<А
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0К
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0I
conv2d/Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:k
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€m
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Х
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskt
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         И
conv2d/Conv2D/ReshapeReshapeinputs$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€Ш
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ќ
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
r
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   d
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ј
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Ю
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@o
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:w
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: В
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€y
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskА
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   Ї
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@¶
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@~
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   p
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€р
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:√
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@
conv2d/ReluRelu,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@У
add/addAddV2(batch_normalization/FusedBatchNormV3:y:0conv2d/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€@b
activation/ReluReluadd/add:z:0*
T0*3
_output_shapes!
:€€€€€€€€€@њ
max_pooling3d/MaxPool3D	MaxPool3Dactivation/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€@*
ksize	
*
paddingVALID*
strides	
e
conv2d_3/Conv2D/ShapeShape max_pooling3d/MaxPool3D:output:0*
T0*
_output_shapes
:m
#conv2d_3/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_3/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_3/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_3/Conv2D/strided_sliceStridedSliceconv2d_3/Conv2D/Shape:output:0,conv2d_3/Conv2D/strided_slice/stack:output:0.conv2d_3/Conv2D/strided_slice/stack_1:output:0.conv2d_3/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_3/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   ¶
conv2d_3/Conv2D/ReshapeReshape max_pooling3d/MaxPool3D:output:0&conv2d_3/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Э
%conv2d_3/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0‘
conv2d_3/Conv2D/Conv2DConv2D conv2d_3/Conv2D/Reshape:output:0-conv2d_3/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
t
conv2d_3/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   f
conv2d_3/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_3/Conv2D/concatConcatV2&conv2d_3/Conv2D/strided_slice:output:0(conv2d_3/Conv2D/concat/values_1:output:0$conv2d_3/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:•
conv2d_3/Conv2D/Reshape_1Reshapeconv2d_3/Conv2D/Conv2D:output:0conv2d_3/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
!conv2d_3/squeeze_batch_dims/ShapeShape"conv2d_3/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_3/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_3/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_3/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_3/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_3/squeeze_batch_dims/Shape:output:08conv2d_3/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_3/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_3/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_3/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   Ѕ
#conv2d_3/squeeze_batch_dims/ReshapeReshape"conv2d_3/Conv2D/Reshape_1:output:02conv2d_3/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_3_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0”
#conv2d_3/squeeze_batch_dims/BiasAddBiasAdd,conv2d_3/squeeze_batch_dims/Reshape:output:0:conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АА
+conv2d_3/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   r
'conv2d_3/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_3/squeeze_batch_dims/concatConcatV22conv2d_3/squeeze_batch_dims/strided_slice:output:04conv2d_3/squeeze_batch_dims/concat/values_1:output:00conv2d_3/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
: 
%conv2d_3/squeeze_batch_dims/Reshape_1Reshape,conv2d_3/squeeze_batch_dims/BiasAdd:output:0+conv2d_3/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АД
conv2d_3/ReluRelu.conv2d_3/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АП
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0и
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<И
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Т
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0e
conv2d_2/Conv2D/ShapeShape max_pooling3d/MaxPool3D:output:0*
T0*
_output_shapes
:m
#conv2d_2/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_2/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_2/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_2/Conv2D/strided_sliceStridedSliceconv2d_2/Conv2D/Shape:output:0,conv2d_2/Conv2D/strided_slice/stack:output:0.conv2d_2/Conv2D/strided_slice/stack_1:output:0.conv2d_2/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_2/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   ¶
conv2d_2/Conv2D/ReshapeReshape max_pooling3d/MaxPool3D:output:0&conv2d_2/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Э
%conv2d_2/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0‘
conv2d_2/Conv2D/Conv2DConv2D conv2d_2/Conv2D/Reshape:output:0-conv2d_2/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
t
conv2d_2/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   f
conv2d_2/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_2/Conv2D/concatConcatV2&conv2d_2/Conv2D/strided_slice:output:0(conv2d_2/Conv2D/concat/values_1:output:0$conv2d_2/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:•
conv2d_2/Conv2D/Reshape_1Reshapeconv2d_2/Conv2D/Conv2D:output:0conv2d_2/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
!conv2d_2/squeeze_batch_dims/ShapeShape"conv2d_2/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_2/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_2/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_2/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_2/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_2/squeeze_batch_dims/Shape:output:08conv2d_2/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_2/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_2/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_2/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   Ѕ
#conv2d_2/squeeze_batch_dims/ReshapeReshape"conv2d_2/Conv2D/Reshape_1:output:02conv2d_2/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_2_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0”
#conv2d_2/squeeze_batch_dims/BiasAddBiasAdd,conv2d_2/squeeze_batch_dims/Reshape:output:0:conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АА
+conv2d_2/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   r
'conv2d_2/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_2/squeeze_batch_dims/concatConcatV22conv2d_2/squeeze_batch_dims/strided_slice:output:04conv2d_2/squeeze_batch_dims/concat/values_1:output:00conv2d_2/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
: 
%conv2d_2/squeeze_batch_dims/Reshape_1Reshape,conv2d_2/squeeze_batch_dims/BiasAdd:output:0+conv2d_2/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АД
conv2d_2/ReluRelu.conv2d_2/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АЪ
	add_1/addAddV2*batch_normalization_1/FusedBatchNormV3:y:0conv2d_2/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аg
activation_1/ReluReluadd_1/add:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аƒ
max_pooling3d_1/MaxPool3D	MaxPool3Dactivation_1/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
ksize	
*
paddingVALID*
strides	
g
conv2d_5/Conv2D/ShapeShape"max_pooling3d_1/MaxPool3D:output:0*
T0*
_output_shapes
:m
#conv2d_5/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_5/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_5/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_5/Conv2D/strided_sliceStridedSliceconv2d_5/Conv2D/Shape:output:0,conv2d_5/Conv2D/strided_slice/stack:output:0.conv2d_5/Conv2D/strided_slice/stack_1:output:0.conv2d_5/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_5/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   ©
conv2d_5/Conv2D/ReshapeReshape"max_pooling3d_1/MaxPool3D:output:0&conv2d_5/Conv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЮ
%conv2d_5/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_5_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0‘
conv2d_5/Conv2D/Conv2DConv2D conv2d_5/Conv2D/Reshape:output:0-conv2d_5/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
t
conv2d_5/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_5/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_5/Conv2D/concatConcatV2&conv2d_5/Conv2D/strided_slice:output:0(conv2d_5/Conv2D/concat/values_1:output:0$conv2d_5/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:•
conv2d_5/Conv2D/Reshape_1Reshapeconv2d_5/Conv2D/Conv2D:output:0conv2d_5/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
!conv2d_5/squeeze_batch_dims/ShapeShape"conv2d_5/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_5/squeeze_batch_dims/Shape:output:08conv2d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Ѕ
#conv2d_5/squeeze_batch_dims/ReshapeReshape"conv2d_5/Conv2D/Reshape_1:output:02conv2d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0”
#conv2d_5/squeeze_batch_dims/BiasAddBiasAdd,conv2d_5/squeeze_batch_dims/Reshape:output:0:conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АА
+conv2d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_5/squeeze_batch_dims/concatConcatV22conv2d_5/squeeze_batch_dims/strided_slice:output:04conv2d_5/squeeze_batch_dims/concat/values_1:output:00conv2d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
: 
%conv2d_5/squeeze_batch_dims/Reshape_1Reshape,conv2d_5/squeeze_batch_dims/BiasAdd:output:0+conv2d_5/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АД
conv2d_5/ReluRelu.conv2d_5/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АП
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0и
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_5/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<И
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Т
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0g
conv2d_4/Conv2D/ShapeShape"max_pooling3d_1/MaxPool3D:output:0*
T0*
_output_shapes
:m
#conv2d_4/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_4/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_4/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_4/Conv2D/strided_sliceStridedSliceconv2d_4/Conv2D/Shape:output:0,conv2d_4/Conv2D/strided_slice/stack:output:0.conv2d_4/Conv2D/strided_slice/stack_1:output:0.conv2d_4/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_4/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   ©
conv2d_4/Conv2D/ReshapeReshape"max_pooling3d_1/MaxPool3D:output:0&conv2d_4/Conv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЮ
%conv2d_4/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0‘
conv2d_4/Conv2D/Conv2DConv2D conv2d_4/Conv2D/Reshape:output:0-conv2d_4/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
t
conv2d_4/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_4/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_4/Conv2D/concatConcatV2&conv2d_4/Conv2D/strided_slice:output:0(conv2d_4/Conv2D/concat/values_1:output:0$conv2d_4/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:•
conv2d_4/Conv2D/Reshape_1Reshapeconv2d_4/Conv2D/Conv2D:output:0conv2d_4/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
!conv2d_4/squeeze_batch_dims/ShapeShape"conv2d_4/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_4/squeeze_batch_dims/Shape:output:08conv2d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Ѕ
#conv2d_4/squeeze_batch_dims/ReshapeReshape"conv2d_4/Conv2D/Reshape_1:output:02conv2d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0”
#conv2d_4/squeeze_batch_dims/BiasAddBiasAdd,conv2d_4/squeeze_batch_dims/Reshape:output:0:conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АА
+conv2d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_4/squeeze_batch_dims/concatConcatV22conv2d_4/squeeze_batch_dims/strided_slice:output:04conv2d_4/squeeze_batch_dims/concat/values_1:output:00conv2d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
: 
%conv2d_4/squeeze_batch_dims/Reshape_1Reshape,conv2d_4/squeeze_batch_dims/BiasAdd:output:0+conv2d_4/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АД
conv2d_4/ReluRelu.conv2d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АЪ
	add_2/addAddV2*batch_normalization_2/FusedBatchNormV3:y:0conv2d_4/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аg
activation_2/ReluReluadd_2/add:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аƒ
max_pooling3d_2/MaxPool3D	MaxPool3Dactivation_2/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
ksize	
*
paddingVALID*
strides	
g
conv2d_7/Conv2D/ShapeShape"max_pooling3d_2/MaxPool3D:output:0*
T0*
_output_shapes
:m
#conv2d_7/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_7/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_7/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_7/Conv2D/strided_sliceStridedSliceconv2d_7/Conv2D/Shape:output:0,conv2d_7/Conv2D/strided_slice/stack:output:0.conv2d_7/Conv2D/strided_slice/stack_1:output:0.conv2d_7/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_7/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ©
conv2d_7/Conv2D/ReshapeReshape"max_pooling3d_2/MaxPool3D:output:0&conv2d_7/Conv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЮ
%conv2d_7/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_7_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0‘
conv2d_7/Conv2D/Conv2DConv2D conv2d_7/Conv2D/Reshape:output:0-conv2d_7/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
t
conv2d_7/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_7/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_7/Conv2D/concatConcatV2&conv2d_7/Conv2D/strided_slice:output:0(conv2d_7/Conv2D/concat/values_1:output:0$conv2d_7/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:•
conv2d_7/Conv2D/Reshape_1Reshapeconv2d_7/Conv2D/Conv2D:output:0conv2d_7/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
!conv2d_7/squeeze_batch_dims/ShapeShape"conv2d_7/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_7/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_7/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_7/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_7/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_7/squeeze_batch_dims/Shape:output:08conv2d_7/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_7/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_7/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_7/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Ѕ
#conv2d_7/squeeze_batch_dims/ReshapeReshape"conv2d_7/Conv2D/Reshape_1:output:02conv2d_7/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
2conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_7_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0”
#conv2d_7/squeeze_batch_dims/BiasAddBiasAdd,conv2d_7/squeeze_batch_dims/Reshape:output:0:conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АА
+conv2d_7/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_7/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_7/squeeze_batch_dims/concatConcatV22conv2d_7/squeeze_batch_dims/strided_slice:output:04conv2d_7/squeeze_batch_dims/concat/values_1:output:00conv2d_7/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
: 
%conv2d_7/squeeze_batch_dims/Reshape_1Reshape,conv2d_7/squeeze_batch_dims/BiasAdd:output:0+conv2d_7/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АД
conv2d_7/ReluRelu.conv2d_7/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АП
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0и
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<И
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Т
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0g
conv2d_6/Conv2D/ShapeShape"max_pooling3d_2/MaxPool3D:output:0*
T0*
_output_shapes
:m
#conv2d_6/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv2d_6/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€o
%conv2d_6/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
conv2d_6/Conv2D/strided_sliceStridedSliceconv2d_6/Conv2D/Shape:output:0,conv2d_6/Conv2D/strided_slice/stack:output:0.conv2d_6/Conv2D/strided_slice/stack_1:output:0.conv2d_6/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv2d_6/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ©
conv2d_6/Conv2D/ReshapeReshape"max_pooling3d_2/MaxPool3D:output:0&conv2d_6/Conv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЮ
%conv2d_6/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_6_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0‘
conv2d_6/Conv2D/Conv2DConv2D conv2d_6/Conv2D/Reshape:output:0-conv2d_6/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
t
conv2d_6/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         f
conv2d_6/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€»
conv2d_6/Conv2D/concatConcatV2&conv2d_6/Conv2D/strided_slice:output:0(conv2d_6/Conv2D/concat/values_1:output:0$conv2d_6/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:•
conv2d_6/Conv2D/Reshape_1Reshapeconv2d_6/Conv2D/Conv2D:output:0conv2d_6/Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аs
!conv2d_6/squeeze_batch_dims/ShapeShape"conv2d_6/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:y
/conv2d_6/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
1conv2d_6/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€{
1conv2d_6/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
)conv2d_6/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_6/squeeze_batch_dims/Shape:output:08conv2d_6/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_6/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_6/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskВ
)conv2d_6/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Ѕ
#conv2d_6/squeeze_batch_dims/ReshapeReshape"conv2d_6/Conv2D/Reshape_1:output:02conv2d_6/squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
2conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_6_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0”
#conv2d_6/squeeze_batch_dims/BiasAddBiasAdd,conv2d_6/squeeze_batch_dims/Reshape:output:0:conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АА
+conv2d_6/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         r
'conv2d_6/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ш
"conv2d_6/squeeze_batch_dims/concatConcatV22conv2d_6/squeeze_batch_dims/strided_slice:output:04conv2d_6/squeeze_batch_dims/concat/values_1:output:00conv2d_6/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
: 
%conv2d_6/squeeze_batch_dims/Reshape_1Reshape,conv2d_6/squeeze_batch_dims/BiasAdd:output:0+conv2d_6/squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АД
conv2d_6/ReluRelu.conv2d_6/squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€АЪ
	add_3/addAddV2*batch_normalization_3/FusedBatchNormV3:y:0conv2d_6/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аg
activation_3/ReluReluadd_3/add:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€А^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ж
flatten/ReshapeReshapeactivation_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аy
v/MatMul/ReadVariableOpReadVariableOp v_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0}
v/MatMulMatMuldense/BiasAdd:output:0v/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€v
v/BiasAdd/ReadVariableOpReadVariableOp!v_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
	v/BiasAddBiasAddv/MatMul:product:0 v/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T
v/TanhTanhv/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€z
p/MatMul/ReadVariableOpReadVariableOp p_matmul_readvariableop_resource* 
_output_shapes
:
А∞!*
dtype0А
p/MatMulMatMulflatten/Reshape:output:0p/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€∞!w
p/BiasAdd/ReadVariableOpReadVariableOp!p_biasadd_readvariableop_resource*
_output_shapes	
:∞!*
dtype0}
	p/BiasAddBiasAddp/MatMul:product:0 p/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€∞![
	p/SoftmaxSoftmaxp/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€∞!c
IdentityIdentityp/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞![

Identity_1Identity
v/Tanh:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Н
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_2/Conv2D/Conv2D/ReadVariableOp3^conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_3/Conv2D/Conv2D/ReadVariableOp3^conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_4/Conv2D/Conv2D/ReadVariableOp3^conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_5/Conv2D/Conv2D/ReadVariableOp3^conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_6/Conv2D/Conv2D/ReadVariableOp3^conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_7/Conv2D/Conv2D/ReadVariableOp3^conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^p/BiasAdd/ReadVariableOp^p/MatMul/ReadVariableOp^v/BiasAdd/ReadVariableOp^v/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_2/Conv2D/Conv2D/ReadVariableOp%conv2d_2/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_2/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_3/Conv2D/Conv2D/ReadVariableOp%conv2d_3/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_4/Conv2D/Conv2D/ReadVariableOp%conv2d_4/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_5/Conv2D/Conv2D/ReadVariableOp%conv2d_5/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_6/Conv2D/Conv2D/ReadVariableOp%conv2d_6/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_6/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_7/Conv2D/Conv2D/ReadVariableOp%conv2d_7/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_7/squeeze_batch_dims/BiasAdd/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp24
p/BiasAdd/ReadVariableOpp/BiasAdd/ReadVariableOp22
p/MatMul/ReadVariableOpp/MatMul/ReadVariableOp24
v/BiasAdd/ReadVariableOpv/BiasAdd/ReadVariableOp22
v/MatMul/ReadVariableOpv/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
Б
e
I__inference_activation_1_layer_call_and_return_conditional_losses_3337583

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :€€€€€€€€€Аg
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
™
°
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3337331

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0т
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ш
Ы
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3339668

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
is_training( К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
≈
Ч
'__inference_dense_layer_call_fn_3340304

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3337809p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ѕ	
÷
7__inference_batch_normalization_1_layer_call_fn_3339815

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3337179Ч
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
–
`
D__inference_flatten_layer_call_and_return_conditional_losses_3340295

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
Е
l
@__inference_add_layer_call_and_return_conditional_losses_3339740
inputs_0
inputs_1
identity^
addAddV2inputs_0inputs_1*
T0*3
_output_shapes!
:€€€€€€€€€@[
IdentityIdentityadd:z:0*
T0*3
_output_shapes!
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€@:€€€€€€€€€@:] Y
3
_output_shapes!
:€€€€€€€€€@
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:€€€€€€€€€@
"
_user_specified_name
inputs/1
і%
µ
E__inference_conv2d_4_layer_call_and_return_conditional_losses_3340084

inputsA
%conv2d_conv2d_readvariableop_resource:ААA
2squeeze_batch_dims_biasadd_readvariableop_resource:	А
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   {
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АМ
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:К
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аa
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ¶
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЩ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Є
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аw
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АС
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
√	
–
5__inference_batch_normalization_layer_call_fn_3339637

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3337103Ц
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ґ%
∞
C__inference_conv2d_layer_call_and_return_conditional_losses_3337461

inputs?
%conv2d_conv2d_readvariableop_resource:@@
2squeeze_batch_dims_biasadd_readvariableop_resource:@
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   •
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Ш
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ј
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@q
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€@С
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
Ѕ	
–
5__inference_batch_normalization_layer_call_fn_3339650

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3337134Ц
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Е
l
B__inference_add_3_layer_call_and_return_conditional_losses_3337782

inputs
inputs_1
identity]
addAddV2inputsinputs_1*
T0*4
_output_shapes"
 :€€€€€€€€€А\
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
А
°
*__inference_conv2d_2_layer_call_fn_3339873

inputs"
unknown:@А
	unknown_0:	А
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_3337564|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
ы
c
G__inference_activation_layer_call_and_return_conditional_losses_3337480

inputs
identityR
ReluReluinputs*
T0*3
_output_shapes!
:€€€€€€€€€@f
IdentityIdentityRelu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
Г
Ґ
*__inference_conv2d_4_layer_call_fn_3340051

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_3337667|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
Ќ	
÷
7__inference_batch_normalization_1_layer_call_fn_3339828

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3337210Ч
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
љ
У
#__inference_p_layer_call_fn_3340323

inputs
unknown:
А∞!
	unknown_0:	∞!
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€∞!*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_p_layer_call_and_return_conditional_losses_3337843p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞!`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
к
K
/__inference_max_pooling3d_layer_call_fn_3339755

inputs
identityе
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_3337154Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
і%
µ
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3339980

inputsA
%conv2d_conv2d_readvariableop_resource:ААA
2squeeze_batch_dims_biasadd_readvariableop_resource:	А
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   {
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АМ
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:К
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аa
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ¶
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЩ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Є
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аw
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АС
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
і%
µ
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3337770

inputsA
%conv2d_conv2d_readvariableop_resource:ААA
2squeeze_batch_dims_biasadd_readvariableop_resource:	А
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         {
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АМ
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:К
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аa
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ¶
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЩ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Є
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аw
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АС
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
і%
µ
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3340158

inputsA
%conv2d_conv2d_readvariableop_resource:ААA
2squeeze_batch_dims_biasadd_readvariableop_resource:	А
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         {
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АМ
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:К
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аa
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ¶
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЩ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Є
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аw
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АС
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
Б
e
I__inference_activation_1_layer_call_and_return_conditional_losses_3339928

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :€€€€€€€€€Аg
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
Є
≈
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3340220

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0А
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ѓ%
і
E__inference_conv2d_2_layer_call_and_return_conditional_losses_3339906

inputs@
%conv2d_conv2d_readvariableop_resource:@АA
2squeeze_batch_dims_biasadd_readvariableop_resource:	А
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Л
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:К
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аa
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   ¶
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЩ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Є
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аw
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      А   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АС
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
Ў
f
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_3339760

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
о
M
1__inference_max_pooling3d_2_layer_call_fn_3340111

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_3337306Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Е
l
B__inference_add_2_layer_call_and_return_conditional_losses_3337679

inputs
inputs_1
identity]
addAddV2inputsinputs_1*
T0*4
_output_shapes"
 :€€€€€€€€€А\
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
Є
≈
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3337210

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0А
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
џ
J
.__inference_activation_2_layer_call_fn_3340101

inputs
identityЅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_3337686m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
§%
≤
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3337413

inputs?
%conv2d_conv2d_readvariableop_resource:@@
2squeeze_batch_dims_biasadd_readvariableop_resource:@
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         z
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@a
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      @   •
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Ш
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ј
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@w
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"      @   i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@q
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€@С
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
ц
ѕ	
'__inference_model_layer_call_fn_3337932
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@#
	unknown_5:@
	unknown_6:@$
	unknown_7:@А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А

unknown_11:	А

unknown_12:	А%

unknown_13:@А

unknown_14:	А&

unknown_15:АА

unknown_16:	А

unknown_17:	А

unknown_18:	А

unknown_19:	А

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А

unknown_25:	А

unknown_26:	А

unknown_27:	А

unknown_28:	А&

unknown_29:АА

unknown_30:	А

unknown_31:
АА

unknown_32:	А

unknown_33:	А

unknown_34:

unknown_35:
А∞!

unknown_36:	∞!
identity

identity_1ИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':€€€€€€€€€∞!:€€€€€€€€€*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_3337851p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞!q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:€€€€€€€€€
!
_user_specified_name	input_1
і%
µ
E__inference_conv2d_4_layer_call_and_return_conditional_losses_3337667

inputsA
%conv2d_conv2d_readvariableop_resource:ААA
2squeeze_batch_dims_biasadd_readvariableop_resource:	А
identityИҐConv2D/Conv2D/ReadVariableOpҐ)squeeze_batch_dims/BiasAdd/ReadVariableOpB
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:d
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€f
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€      А   {
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АМ
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0є
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
k
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         ]
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:К
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аa
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
э€€€€€€€€r
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masky
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ¶
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€АЩ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Є
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аw
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€‘
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ѓ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аr
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АС
NoOpNoOp^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
™
°
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3339846

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0т
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Є
≈
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3337286

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0А
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Џ
h
L__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_3337306

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Б
e
I__inference_activation_2_layer_call_and_return_conditional_losses_3340106

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :€€€€€€€€€Аg
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
Г
Ґ
*__inference_conv2d_5_layer_call_fn_3339947

inputs#
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3337619|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
Џ
h
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_3337230

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Їѕ
≠.
 __inference__traced_save_3340705
file_prefix.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop'
#savev2_p_kernel_read_readvariableop%
!savev2_p_bias_read_readvariableop'
#savev2_v_kernel_read_readvariableop%
!savev2_v_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_7_kernel_m_read_readvariableop3
/savev2_adam_conv2d_7_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop.
*savev2_adam_p_kernel_m_read_readvariableop,
(savev2_adam_p_bias_m_read_readvariableop.
*savev2_adam_v_kernel_m_read_readvariableop,
(savev2_adam_v_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_7_kernel_v_read_readvariableop3
/savev2_adam_conv2d_7_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop.
*savev2_adam_p_kernel_v_read_readvariableop,
(savev2_adam_p_bias_v_read_readvariableop.
*savev2_adam_v_kernel_v_read_readvariableop,
(savev2_adam_v_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: •=
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:n*
dtype0*ќ<
valueƒ<BЅ<nB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHћ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:n*
dtype0*с
valueзBдnB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ©,
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop#savev2_p_kernel_read_readvariableop!savev2_p_bias_read_readvariableop#savev2_v_kernel_read_readvariableop!savev2_v_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_7_kernel_m_read_readvariableop/savev2_adam_conv2d_7_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop*savev2_adam_p_kernel_m_read_readvariableop(savev2_adam_p_bias_m_read_readvariableop*savev2_adam_v_kernel_m_read_readvariableop(savev2_adam_v_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_7_kernel_v_read_readvariableop/savev2_adam_conv2d_7_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop*savev2_adam_p_kernel_v_read_readvariableop(savev2_adam_p_bias_v_read_readvariableop*savev2_adam_v_kernel_v_read_readvariableop(savev2_adam_v_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *|
dtypesr
p2n	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ь
_input_shapesК
З: :@:@:@:@:@:@:@:@:@А:А:А:А:А:А:@А:А:АА:А:А:А:А:А:АА:А:АА:А:А:А:А:А:АА:А:
АА:А:
А∞!:∞!:	А:: : : : : : : : : : : :@:@:@:@:@:@:@А:А:А:А:@А:А:АА:А:А:А:АА:А:АА:А:А:А:АА:А:
АА:А:
А∞!:∞!:	А::@:@:@:@:@:@:@А:А:А:А:@А:А:АА:А:А:А:АА:А:АА:А:А:А:АА:А:
АА:А:
А∞!:∞!:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
:@:-	)
'
_output_shapes
:@А:!


_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:! 

_output_shapes	
:А:&!"
 
_output_shapes
:
АА:!"

_output_shapes	
:А:&#"
 
_output_shapes
:
А∞!:!$

_output_shapes	
:∞!:%%!

_output_shapes
:	А: &

_output_shapes
::'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :,2(
&
_output_shapes
:@: 3

_output_shapes
:@: 4

_output_shapes
:@: 5

_output_shapes
:@:,6(
&
_output_shapes
:@: 7

_output_shapes
:@:-8)
'
_output_shapes
:@А:!9

_output_shapes	
:А:!:

_output_shapes	
:А:!;

_output_shapes	
:А:-<)
'
_output_shapes
:@А:!=

_output_shapes	
:А:.>*
(
_output_shapes
:АА:!?

_output_shapes	
:А:!@

_output_shapes	
:А:!A

_output_shapes	
:А:.B*
(
_output_shapes
:АА:!C

_output_shapes	
:А:.D*
(
_output_shapes
:АА:!E

_output_shapes	
:А:!F

_output_shapes	
:А:!G

_output_shapes	
:А:.H*
(
_output_shapes
:АА:!I

_output_shapes	
:А:&J"
 
_output_shapes
:
АА:!K

_output_shapes	
:А:&L"
 
_output_shapes
:
А∞!:!M

_output_shapes	
:∞!:%N!

_output_shapes
:	А: O

_output_shapes
::,P(
&
_output_shapes
:@: Q

_output_shapes
:@: R

_output_shapes
:@: S

_output_shapes
:@:,T(
&
_output_shapes
:@: U

_output_shapes
:@:-V)
'
_output_shapes
:@А:!W

_output_shapes	
:А:!X

_output_shapes	
:А:!Y

_output_shapes	
:А:-Z)
'
_output_shapes
:@А:![

_output_shapes	
:А:.\*
(
_output_shapes
:АА:!]

_output_shapes	
:А:!^

_output_shapes	
:А:!_

_output_shapes	
:А:.`*
(
_output_shapes
:АА:!a

_output_shapes	
:А:.b*
(
_output_shapes
:АА:!c

_output_shapes	
:А:!d

_output_shapes	
:А:!e

_output_shapes	
:А:.f*
(
_output_shapes
:АА:!g

_output_shapes	
:А:&h"
 
_output_shapes
:
АА:!i

_output_shapes	
:А:&j"
 
_output_shapes
:
А∞!:!k

_output_shapes	
:∞!:%l!

_output_shapes
:	А: m

_output_shapes
::n

_output_shapes
: 
”
H
,__inference_activation_layer_call_fn_3339745

inputs
identityЊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_3337480l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
џ
J
.__inference_activation_3_layer_call_fn_3340279

inputs
identityЅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_3337789m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€А:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
м
S
'__inference_add_1_layer_call_fn_3339912
inputs_0
inputs_1
identity«
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_3337576m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€А:^ Z
4
_output_shapes"
 :€€€€€€€€€А
"
_user_specified_name
inputs/0:^Z
4
_output_shapes"
 :€€€€€€€€€А
"
_user_specified_name
inputs/1
ъy
О
B__inference_model_layer_call_and_return_conditional_losses_3338293

inputs*
conv2d_1_3338188:@
conv2d_1_3338190:@)
batch_normalization_3338193:@)
batch_normalization_3338195:@)
batch_normalization_3338197:@)
batch_normalization_3338199:@(
conv2d_3338202:@
conv2d_3338204:@+
conv2d_3_3338210:@А
conv2d_3_3338212:	А,
batch_normalization_1_3338215:	А,
batch_normalization_1_3338217:	А,
batch_normalization_1_3338219:	А,
batch_normalization_1_3338221:	А+
conv2d_2_3338224:@А
conv2d_2_3338226:	А,
conv2d_5_3338232:АА
conv2d_5_3338234:	А,
batch_normalization_2_3338237:	А,
batch_normalization_2_3338239:	А,
batch_normalization_2_3338241:	А,
batch_normalization_2_3338243:	А,
conv2d_4_3338246:АА
conv2d_4_3338248:	А,
conv2d_7_3338254:АА
conv2d_7_3338256:	А,
batch_normalization_3_3338259:	А,
batch_normalization_3_3338261:	А,
batch_normalization_3_3338263:	А,
batch_normalization_3_3338265:	А,
conv2d_6_3338268:АА
conv2d_6_3338270:	А!
dense_3338276:
АА
dense_3338278:	А
	v_3338281:	А
	v_3338283:
	p_3338286:
А∞!
	p_3338288:	∞!
identity

identity_1ИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐp/StatefulPartitionedCallҐv/StatefulPartitionedCall€
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_3338188conv2d_1_3338190*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3337413К
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_3338193batch_normalization_3338195batch_normalization_3338197batch_normalization_3338199*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3337134ч
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3338202conv2d_3338204*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_3337461У
add/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_3337473я
activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_3337480м
max_pooling3d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_3337154†
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv2d_3_3338210conv2d_3_3338212*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_3337516Ч
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_1_3338215batch_normalization_1_3338217batch_normalization_1_3338219batch_normalization_1_3338221*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3337210†
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv2d_2_3338224conv2d_2_3338226*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_3337564Ь
add_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_3337576ж
activation_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_3337583у
max_pooling3d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_3337230Ґ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv2d_5_3338232conv2d_5_3338234*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3337619Ч
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_2_3338237batch_normalization_2_3338239batch_normalization_2_3338241batch_normalization_2_3338243*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3337286Ґ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv2d_4_3338246conv2d_4_3338248*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_3337667Ь
add_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_3337679ж
activation_2/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_3337686у
max_pooling3d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_3337306Ґ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv2d_7_3338254conv2d_7_3338256*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3337722Ч
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_3_3338259batch_normalization_3_3338261batch_normalization_3_3338263batch_normalization_3_3338265*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3337362Ґ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv2d_6_3338268conv2d_6_3338270*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3337770Ь
add_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_3_layer_call_and_return_conditional_losses_3337782ж
activation_3/PartitionedCallPartitionedCalladd_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_3337789„
flatten/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_3337797В
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_3338276dense_3338278*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_3337809ч
v/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0	v_3338281	v_3338283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_v_layer_call_and_return_conditional_losses_3337826т
p/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	p_3338286	p_3338288*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€∞!*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_p_layer_call_and_return_conditional_losses_3337843r
IdentityIdentity"p/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€∞!s

Identity_1Identity"v/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€т
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall^p/StatefulPartitionedCall^v/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*~
_input_shapesm
k:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall26
p/StatefulPartitionedCallp/StatefulPartitionedCall26
v/StatefulPartitionedCallv/StatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
™
°
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3340202

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0т
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Є
≈
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3337362

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0А
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ш
Ы
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3337103

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
is_training( К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs"џL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*и
serving_default‘
G
input_1<
serving_default_input_1:0€€€€€€€€€6
p1
StatefulPartitionedCall:0€€€€€€€€€∞!5
v0
StatefulPartitionedCall:1€€€€€€€€€tensorflow/serving/predict:®а
Ы
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer-17
layer-18
layer_with_weights-9
layer-19
layer_with_weights-10
layer-20
layer_with_weights-11
layer-21
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
layer_with_weights-14
layer-27
	optimizer
loss

signatures
# _self_saveable_object_factories
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_default_save_signature"
_tf_keras_network
D
#(_self_saveable_object_factories"
_tf_keras_input_layer
а

)kernel
*bias
#+_self_saveable_object_factories
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
П
2axis
	3gamma
4beta
5moving_mean
6moving_variance
#7_self_saveable_object_factories
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
а

>kernel
?bias
#@_self_saveable_object_factories
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
 
#G_self_saveable_object_factories
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
 
#N_self_saveable_object_factories
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
 
#U_self_saveable_object_factories
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
а

\kernel
]bias
#^_self_saveable_object_factories
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
П
eaxis
	fgamma
gbeta
hmoving_mean
imoving_variance
#j_self_saveable_object_factories
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_layer
а

qkernel
rbias
#s_self_saveable_object_factories
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
#z_self_saveable_object_factories
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+А&call_and_return_all_conditional_losses"
_tf_keras_layer
—
$Б_self_saveable_object_factories
В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"
_tf_keras_layer
—
$И_self_saveable_object_factories
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"
_tf_keras_layer
й
Пkernel
	Рbias
$С_self_saveable_object_factories
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses"
_tf_keras_layer
Ы
	Шaxis

Щgamma
	Ъbeta
Ыmoving_mean
Ьmoving_variance
$Э_self_saveable_object_factories
Ю	variables
Яtrainable_variables
†regularization_losses
°	keras_api
Ґ__call__
+£&call_and_return_all_conditional_losses"
_tf_keras_layer
й
§kernel
	•bias
$¶_self_saveable_object_factories
І	variables
®trainable_variables
©regularization_losses
™	keras_api
Ђ__call__
+ђ&call_and_return_all_conditional_losses"
_tf_keras_layer
—
$≠_self_saveable_object_factories
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
±	keras_api
≤__call__
+≥&call_and_return_all_conditional_losses"
_tf_keras_layer
—
$і_self_saveable_object_factories
µ	variables
ґtrainable_variables
Јregularization_losses
Є	keras_api
є__call__
+Ї&call_and_return_all_conditional_losses"
_tf_keras_layer
—
$ї_self_saveable_object_factories
Љ	variables
љtrainable_variables
Њregularization_losses
њ	keras_api
ј__call__
+Ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
й
¬kernel
	√bias
$ƒ_self_saveable_object_factories
≈	variables
∆trainable_variables
«regularization_losses
»	keras_api
…__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
Ы
	Ћaxis

ћgamma
	Ќbeta
ќmoving_mean
ѕmoving_variance
$–_self_saveable_object_factories
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
’__call__
+÷&call_and_return_all_conditional_losses"
_tf_keras_layer
й
„kernel
	Ўbias
$ў_self_saveable_object_factories
Џ	variables
џtrainable_variables
№regularization_losses
Ё	keras_api
ё__call__
+я&call_and_return_all_conditional_losses"
_tf_keras_layer
—
$а_self_saveable_object_factories
б	variables
вtrainable_variables
гregularization_losses
д	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"
_tf_keras_layer
—
$з_self_saveable_object_factories
и	variables
йtrainable_variables
кregularization_losses
л	keras_api
м__call__
+н&call_and_return_all_conditional_losses"
_tf_keras_layer
—
$о_self_saveable_object_factories
п	variables
рtrainable_variables
сregularization_losses
т	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"
_tf_keras_layer
й
хkernel
	цbias
$ч_self_saveable_object_factories
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses"
_tf_keras_layer
й
юkernel
	€bias
$А_self_saveable_object_factories
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"
_tf_keras_layer
й
Зkernel
	Иbias
$Й_self_saveable_object_factories
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses"
_tf_keras_layer
‘
	Рiter
Сbeta_1
Тbeta_2

Уdecay
Фlearning_rate)m±*m≤3m≥4mі>mµ?mґ\mЈ]mЄfmєgmЇqmїrmЉ	Пmљ	РmЊ	Щmњ	Ъmј	§mЅ	•m¬	¬m√	√mƒ	ћm≈	Ќm∆	„m«	Ўm»	хm…	цm 	юmЋ	€mћ	ЗmЌ	Иmќ)vѕ*v–3v—4v“>v”?v‘\v’]v÷fv„gvЎqvўrvЏ	Пvџ	Рv№	ЩvЁ	Ъvё	§vя	•vа	¬vб	√vв	ћvг	Ќvд	„vе	Ўvж	хvз	цvи	юvй	€vк	Зvл	Иvм"
	optimizer
 "
trackable_dict_wrapper
-
Хserving_default"
signature_map
 "
trackable_dict_wrapper
№
)0
*1
32
43
54
65
>6
?7
\8
]9
f10
g11
h12
i13
q14
r15
П16
Р17
Щ18
Ъ19
Ы20
Ь21
§22
•23
¬24
√25
ћ26
Ќ27
ќ28
ѕ29
„30
Ў31
х32
ц33
ю34
€35
З36
И37"
trackable_list_wrapper
Ш
)0
*1
32
43
>4
?5
\6
]7
f8
g9
q10
r11
П12
Р13
Щ14
Ъ15
§16
•17
¬18
√19
ћ20
Ќ21
„22
Ў23
х24
ц25
ю26
€27
З28
И29"
trackable_list_wrapper
 "
trackable_list_wrapper
ѕ
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
'_default_save_signature
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
к2з
'__inference_model_layer_call_fn_3337932
'__inference_model_layer_call_fn_3338762
'__inference_model_layer_call_fn_3338845
'__inference_model_layer_call_fn_3338457ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
B__inference_model_layer_call_and_return_conditional_losses_3339171
B__inference_model_layer_call_and_return_conditional_losses_3339497
B__inference_model_layer_call_and_return_conditional_losses_3338565
B__inference_model_layer_call_and_return_conditional_losses_3338673ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЌB 
"__inference__wrapped_model_3337081input_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
):'@2conv2d_1/kernel
:@2conv2d_1/bias
 "
trackable_dict_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
‘2—
*__inference_conv2d_1_layer_call_fn_3339591Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3339624Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
 "
trackable_dict_wrapper
<
30
41
52
63"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
®2•
5__inference_batch_normalization_layer_call_fn_3339637
5__inference_batch_normalization_layer_call_fn_3339650і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ё2џ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3339668
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3339686і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
':%@2conv2d/kernel
:@2conv2d/bias
 "
trackable_dict_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
“2ѕ
(__inference_conv2d_layer_call_fn_3339695Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_conv2d_layer_call_and_return_conditional_losses_3339728Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
™non_trainable_variables
Ђlayers
ђmetrics
 ≠layer_regularization_losses
Ѓlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
ѕ2ћ
%__inference_add_layer_call_fn_3339734Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
@__inference_add_layer_call_and_return_conditional_losses_3339740Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
ѓnon_trainable_variables
∞layers
±metrics
 ≤layer_regularization_losses
≥layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
÷2”
,__inference_activation_layer_call_fn_3339745Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_activation_layer_call_and_return_conditional_losses_3339750Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
ў2÷
/__inference_max_pooling3d_layer_call_fn_3339755Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ф2с
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_3339760Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
*:(@А2conv2d_3/kernel
:А2conv2d_3/bias
 "
trackable_dict_wrapper
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
єnon_trainable_variables
Їlayers
їmetrics
 Љlayer_regularization_losses
љlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
‘2—
*__inference_conv2d_3_layer_call_fn_3339769Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_conv2d_3_layer_call_and_return_conditional_losses_3339802Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
*:(А2batch_normalization_1/gamma
):'А2batch_normalization_1/beta
2:0А (2!batch_normalization_1/moving_mean
6:4А (2%batch_normalization_1/moving_variance
 "
trackable_dict_wrapper
<
f0
g1
h2
i3"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Њnon_trainable_variables
њlayers
јmetrics
 Ѕlayer_regularization_losses
¬layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
ђ2©
7__inference_batch_normalization_1_layer_call_fn_3339815
7__inference_batch_normalization_1_layer_call_fn_3339828і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3339846
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3339864і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
*:(@А2conv2d_2/kernel
:А2conv2d_2/bias
 "
trackable_dict_wrapper
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
‘2—
*__inference_conv2d_2_layer_call_fn_3339873Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_conv2d_2_layer_call_and_return_conditional_losses_3339906Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
і
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
—2ќ
'__inference_add_1_layer_call_fn_3339912Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_add_1_layer_call_and_return_conditional_losses_3339918Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
Ў2’
.__inference_activation_1_layer_call_fn_3339923Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_activation_1_layer_call_and_return_conditional_losses_3339928Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
Й	variables
Кtrainable_variables
Лregularization_losses
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
џ2Ў
1__inference_max_pooling3d_1_layer_call_fn_3339933Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ц2у
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_3339938Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
+:)АА2conv2d_5/kernel
:А2conv2d_5/bias
 "
trackable_dict_wrapper
0
П0
Р1"
trackable_list_wrapper
0
П0
Р1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
‘2—
*__inference_conv2d_5_layer_call_fn_3339947Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3339980Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
*:(А2batch_normalization_2/gamma
):'А2batch_normalization_2/beta
2:0А (2!batch_normalization_2/moving_mean
6:4А (2%batch_normalization_2/moving_variance
 "
trackable_dict_wrapper
@
Щ0
Ъ1
Ы2
Ь3"
trackable_list_wrapper
0
Щ0
Ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
№non_trainable_variables
Ёlayers
ёmetrics
 яlayer_regularization_losses
аlayer_metrics
Ю	variables
Яtrainable_variables
†regularization_losses
Ґ__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
ђ2©
7__inference_batch_normalization_2_layer_call_fn_3339993
7__inference_batch_normalization_2_layer_call_fn_3340006і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3340024
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3340042і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
+:)АА2conv2d_4/kernel
:А2conv2d_4/bias
 "
trackable_dict_wrapper
0
§0
•1"
trackable_list_wrapper
0
§0
•1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
І	variables
®trainable_variables
©regularization_losses
Ђ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
‘2—
*__inference_conv2d_4_layer_call_fn_3340051Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_conv2d_4_layer_call_and_return_conditional_losses_3340084Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
≤__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses"
_generic_user_object
—2ќ
'__inference_add_2_layer_call_fn_3340090Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_add_2_layer_call_and_return_conditional_losses_3340096Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
µ	variables
ґtrainable_variables
Јregularization_losses
є__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
Ў2’
.__inference_activation_2_layer_call_fn_3340101Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_activation_2_layer_call_and_return_conditional_losses_3340106Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
Љ	variables
љtrainable_variables
Њregularization_losses
ј__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
џ2Ў
1__inference_max_pooling3d_2_layer_call_fn_3340111Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ц2у
L__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_3340116Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
+:)АА2conv2d_7/kernel
:А2conv2d_7/bias
 "
trackable_dict_wrapper
0
¬0
√1"
trackable_list_wrapper
0
¬0
√1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
≈	variables
∆trainable_variables
«regularization_losses
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
‘2—
*__inference_conv2d_7_layer_call_fn_3340125Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3340158Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
*:(А2batch_normalization_3/gamma
):'А2batch_normalization_3/beta
2:0А (2!batch_normalization_3/moving_mean
6:4А (2%batch_normalization_3/moving_variance
 "
trackable_dict_wrapper
@
ћ0
Ќ1
ќ2
ѕ3"
trackable_list_wrapper
0
ћ0
Ќ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
—	variables
“trainable_variables
”regularization_losses
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
ђ2©
7__inference_batch_normalization_3_layer_call_fn_3340171
7__inference_batch_normalization_3_layer_call_fn_3340184і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3340202
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3340220і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
+:)АА2conv2d_6/kernel
:А2conv2d_6/bias
 "
trackable_dict_wrapper
0
„0
Ў1"
trackable_list_wrapper
0
„0
Ў1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
€non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
Џ	variables
џtrainable_variables
№regularization_losses
ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
‘2—
*__inference_conv2d_6_layer_call_fn_3340229Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3340262Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
б	variables
вtrainable_variables
гregularization_losses
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
—2ќ
'__inference_add_3_layer_call_fn_3340268Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_add_3_layer_call_and_return_conditional_losses_3340274Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
и	variables
йtrainable_variables
кregularization_losses
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
Ў2’
.__inference_activation_3_layer_call_fn_3340279Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_activation_3_layer_call_and_return_conditional_losses_3340284Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
п	variables
рtrainable_variables
сregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
”2–
)__inference_flatten_layer_call_fn_3340289Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_flatten_layer_call_and_return_conditional_losses_3340295Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 :
АА2dense/kernel
:А2
dense/bias
 "
trackable_dict_wrapper
0
х0
ц1"
trackable_list_wrapper
0
х0
ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
—2ќ
'__inference_dense_layer_call_fn_3340304Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_layer_call_and_return_conditional_losses_3340314Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
:
А∞!2p/kernel
:∞!2p/bias
 "
trackable_dict_wrapper
0
ю0
€1"
trackable_list_wrapper
0
ю0
€1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
Ќ2 
#__inference_p_layer_call_fn_3340323Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
и2е
>__inference_p_layer_call_and_return_conditional_losses_3340334Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
:	А2v/kernel
:2v/bias
 "
trackable_dict_wrapper
0
З0
И1"
trackable_list_wrapper
0
З0
И1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
Ќ2 
#__inference_v_layer_call_fn_3340343Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
и2е
>__inference_v_layer_call_and_return_conditional_losses_3340354Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ћB…
%__inference_signature_wrapper_3339582input_1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
\
50
61
h2
i3
Ы4
Ь5
ќ6
ѕ7"
trackable_list_wrapper
ц
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27"
trackable_list_wrapper
8
Ґ0
£1
§2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
ќ0
ѕ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

•total

¶count
І	variables
®	keras_api"
_tf_keras_metric
R

©total

™count
Ђ	variables
ђ	keras_api"
_tf_keras_metric
R

≠total

Ѓcount
ѓ	variables
∞	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
•0
¶1"
trackable_list_wrapper
.
І	variables"
_generic_user_object
:  (2total
:  (2count
0
©0
™1"
trackable_list_wrapper
.
Ђ	variables"
_generic_user_object
:  (2total
:  (2count
0
≠0
Ѓ1"
trackable_list_wrapper
.
ѓ	variables"
_generic_user_object
.:,@2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
,:*@2 Adam/batch_normalization/gamma/m
+:)@2Adam/batch_normalization/beta/m
,:*@2Adam/conv2d/kernel/m
:@2Adam/conv2d/bias/m
/:-@А2Adam/conv2d_3/kernel/m
!:А2Adam/conv2d_3/bias/m
/:-А2"Adam/batch_normalization_1/gamma/m
.:,А2!Adam/batch_normalization_1/beta/m
/:-@А2Adam/conv2d_2/kernel/m
!:А2Adam/conv2d_2/bias/m
0:.АА2Adam/conv2d_5/kernel/m
!:А2Adam/conv2d_5/bias/m
/:-А2"Adam/batch_normalization_2/gamma/m
.:,А2!Adam/batch_normalization_2/beta/m
0:.АА2Adam/conv2d_4/kernel/m
!:А2Adam/conv2d_4/bias/m
0:.АА2Adam/conv2d_7/kernel/m
!:А2Adam/conv2d_7/bias/m
/:-А2"Adam/batch_normalization_3/gamma/m
.:,А2!Adam/batch_normalization_3/beta/m
0:.АА2Adam/conv2d_6/kernel/m
!:А2Adam/conv2d_6/bias/m
%:#
АА2Adam/dense/kernel/m
:А2Adam/dense/bias/m
!:
А∞!2Adam/p/kernel/m
:∞!2Adam/p/bias/m
 :	А2Adam/v/kernel/m
:2Adam/v/bias/m
.:,@2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
,:*@2 Adam/batch_normalization/gamma/v
+:)@2Adam/batch_normalization/beta/v
,:*@2Adam/conv2d/kernel/v
:@2Adam/conv2d/bias/v
/:-@А2Adam/conv2d_3/kernel/v
!:А2Adam/conv2d_3/bias/v
/:-А2"Adam/batch_normalization_1/gamma/v
.:,А2!Adam/batch_normalization_1/beta/v
/:-@А2Adam/conv2d_2/kernel/v
!:А2Adam/conv2d_2/bias/v
0:.АА2Adam/conv2d_5/kernel/v
!:А2Adam/conv2d_5/bias/v
/:-А2"Adam/batch_normalization_2/gamma/v
.:,А2!Adam/batch_normalization_2/beta/v
0:.АА2Adam/conv2d_4/kernel/v
!:А2Adam/conv2d_4/bias/v
0:.АА2Adam/conv2d_7/kernel/v
!:А2Adam/conv2d_7/bias/v
/:-А2"Adam/batch_normalization_3/gamma/v
.:,А2!Adam/batch_normalization_3/beta/v
0:.АА2Adam/conv2d_6/kernel/v
!:А2Adam/conv2d_6/bias/v
%:#
АА2Adam/dense/kernel/v
:А2Adam/dense/bias/v
!:
А∞!2Adam/p/kernel/v
:∞!2Adam/p/bias/v
 :	А2Adam/v/kernel/v
:2Adam/v/bias/vн
"__inference__wrapped_model_3337081∆<)*3456>?\]fghiqrПРЩЪЫЬ§•¬√ћЌќѕ„ЎхцЗИю€<Ґ9
2Ґ/
-К*
input_1€€€€€€€€€
™ "H™E
!
pК
p€€€€€€€€€∞!
 
vК
v€€€€€€€€€њ
I__inference_activation_1_layer_call_and_return_conditional_losses_3339928r<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "2Ґ/
(К%
0€€€€€€€€€А
Ъ Ч
.__inference_activation_1_layer_call_fn_3339923e<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "%К"€€€€€€€€€Ањ
I__inference_activation_2_layer_call_and_return_conditional_losses_3340106r<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "2Ґ/
(К%
0€€€€€€€€€А
Ъ Ч
.__inference_activation_2_layer_call_fn_3340101e<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "%К"€€€€€€€€€Ањ
I__inference_activation_3_layer_call_and_return_conditional_losses_3340284r<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "2Ґ/
(К%
0€€€€€€€€€А
Ъ Ч
.__inference_activation_3_layer_call_fn_3340279e<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "%К"€€€€€€€€€Аї
G__inference_activation_layer_call_and_return_conditional_losses_3339750p;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ "1Ґ.
'К$
0€€€€€€€€€@
Ъ У
,__inference_activation_layer_call_fn_3339745c;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ "$К!€€€€€€€€€@с
B__inference_add_1_layer_call_and_return_conditional_losses_3339918™tҐq
jҐg
eЪb
/К,
inputs/0€€€€€€€€€А
/К,
inputs/1€€€€€€€€€А
™ "2Ґ/
(К%
0€€€€€€€€€А
Ъ …
'__inference_add_1_layer_call_fn_3339912ЭtҐq
jҐg
eЪb
/К,
inputs/0€€€€€€€€€А
/К,
inputs/1€€€€€€€€€А
™ "%К"€€€€€€€€€Ас
B__inference_add_2_layer_call_and_return_conditional_losses_3340096™tҐq
jҐg
eЪb
/К,
inputs/0€€€€€€€€€А
/К,
inputs/1€€€€€€€€€А
™ "2Ґ/
(К%
0€€€€€€€€€А
Ъ …
'__inference_add_2_layer_call_fn_3340090ЭtҐq
jҐg
eЪb
/К,
inputs/0€€€€€€€€€А
/К,
inputs/1€€€€€€€€€А
™ "%К"€€€€€€€€€Ас
B__inference_add_3_layer_call_and_return_conditional_losses_3340274™tҐq
jҐg
eЪb
/К,
inputs/0€€€€€€€€€А
/К,
inputs/1€€€€€€€€€А
™ "2Ґ/
(К%
0€€€€€€€€€А
Ъ …
'__inference_add_3_layer_call_fn_3340268ЭtҐq
jҐg
eЪb
/К,
inputs/0€€€€€€€€€А
/К,
inputs/1€€€€€€€€€А
™ "%К"€€€€€€€€€Ам
@__inference_add_layer_call_and_return_conditional_losses_3339740ІrҐo
hҐe
cЪ`
.К+
inputs/0€€€€€€€€€@
.К+
inputs/1€€€€€€€€€@
™ "1Ґ.
'К$
0€€€€€€€€€@
Ъ ƒ
%__inference_add_layer_call_fn_3339734ЪrҐo
hҐe
cЪ`
.К+
inputs/0€€€€€€€€€@
.К+
inputs/1€€€€€€€€€@
™ "$К!€€€€€€€€€@Й
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3339846≤fghi[ҐX
QҐN
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "MҐJ
CК@
09€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Й
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3339864≤fghi[ҐX
QҐN
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "MҐJ
CК@
09€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ б
7__inference_batch_normalization_1_layer_call_fn_3339815•fghi[ҐX
QҐN
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@К=9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Аб
7__inference_batch_normalization_1_layer_call_fn_3339828•fghi[ҐX
QҐN
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@К=9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€АН
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3340024ґЩЪЫЬ[ҐX
QҐN
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "MҐJ
CК@
09€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Н
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3340042ґЩЪЫЬ[ҐX
QҐN
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "MҐJ
CК@
09€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ е
7__inference_batch_normalization_2_layer_call_fn_3339993©ЩЪЫЬ[ҐX
QҐN
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@К=9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ае
7__inference_batch_normalization_2_layer_call_fn_3340006©ЩЪЫЬ[ҐX
QҐN
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@К=9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€АН
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3340202ґћЌќѕ[ҐX
QҐN
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "MҐJ
CК@
09€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Н
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3340220ґћЌќѕ[ҐX
QҐN
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "MҐJ
CК@
09€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ е
7__inference_batch_normalization_3_layer_call_fn_3340171©ћЌќѕ[ҐX
QҐN
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@К=9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ае
7__inference_batch_normalization_3_layer_call_fn_3340184©ћЌќѕ[ҐX
QҐN
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@К=9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€АЕ
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3339668∞3456ZҐW
PҐM
GКD
inputs8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "LҐI
BК?
08€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ Е
P__inference_batch_normalization_layer_call_and_return_conditional_losses_3339686∞3456ZҐW
PҐM
GКD
inputs8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "LҐI
BК?
08€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ Ё
5__inference_batch_normalization_layer_call_fn_3339637£3456ZҐW
PҐM
GКD
inputs8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?К<8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ё
5__inference_batch_normalization_layer_call_fn_3339650£3456ZҐW
PҐM
GКD
inputs8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?К<8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@љ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3339624t)*;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€
™ "1Ґ.
'К$
0€€€€€€€€€@
Ъ Х
*__inference_conv2d_1_layer_call_fn_3339591g)*;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€
™ "$К!€€€€€€€€€@Њ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_3339906uqr;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ "2Ґ/
(К%
0€€€€€€€€€А
Ъ Ц
*__inference_conv2d_2_layer_call_fn_3339873hqr;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ "%К"€€€€€€€€€АЊ
E__inference_conv2d_3_layer_call_and_return_conditional_losses_3339802u\];Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ "2Ґ/
(К%
0€€€€€€€€€А
Ъ Ц
*__inference_conv2d_3_layer_call_fn_3339769h\];Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ "%К"€€€€€€€€€АЅ
E__inference_conv2d_4_layer_call_and_return_conditional_losses_3340084x§•<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "2Ґ/
(К%
0€€€€€€€€€А
Ъ Щ
*__inference_conv2d_4_layer_call_fn_3340051k§•<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "%К"€€€€€€€€€АЅ
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3339980xПР<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "2Ґ/
(К%
0€€€€€€€€€А
Ъ Щ
*__inference_conv2d_5_layer_call_fn_3339947kПР<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "%К"€€€€€€€€€АЅ
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3340262x„Ў<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "2Ґ/
(К%
0€€€€€€€€€А
Ъ Щ
*__inference_conv2d_6_layer_call_fn_3340229k„Ў<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "%К"€€€€€€€€€АЅ
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3340158x¬√<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "2Ґ/
(К%
0€€€€€€€€€А
Ъ Щ
*__inference_conv2d_7_layer_call_fn_3340125k¬√<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "%К"€€€€€€€€€Аї
C__inference_conv2d_layer_call_and_return_conditional_losses_3339728t>?;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€
™ "1Ґ.
'К$
0€€€€€€€€€@
Ъ У
(__inference_conv2d_layer_call_fn_3339695g>?;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€
™ "$К!€€€€€€€€€@¶
B__inference_dense_layer_call_and_return_conditional_losses_3340314`хц0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ~
'__inference_dense_layer_call_fn_3340304Sхц0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€АЃ
D__inference_flatten_layer_call_and_return_conditional_losses_3340295f<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ Ж
)__inference_flatten_layer_call_fn_3340289Y<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "К€€€€€€€€€АЙ
L__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_3339938Є_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "UҐR
KКH
0A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ б
1__inference_max_pooling3d_1_layer_call_fn_3339933Ђ_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HКEA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Й
L__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_3340116Є_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "UҐR
KКH
0A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ б
1__inference_max_pooling3d_2_layer_call_fn_3340111Ђ_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HКEA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€З
J__inference_max_pooling3d_layer_call_and_return_conditional_losses_3339760Є_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "UҐR
KКH
0A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ я
/__inference_max_pooling3d_layer_call_fn_3339755Ђ_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HКEA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Щ
B__inference_model_layer_call_and_return_conditional_losses_3338565“<)*3456>?\]fghiqrПРЩЪЫЬ§•¬√ћЌќѕ„ЎхцЗИю€DҐA
:Ґ7
-К*
input_1€€€€€€€€€
p 

 
™ "LҐI
BЪ?
К
0/0€€€€€€€€€∞!
К
0/1€€€€€€€€€
Ъ Щ
B__inference_model_layer_call_and_return_conditional_losses_3338673“<)*3456>?\]fghiqrПРЩЪЫЬ§•¬√ћЌќѕ„ЎхцЗИю€DҐA
:Ґ7
-К*
input_1€€€€€€€€€
p

 
™ "LҐI
BЪ?
К
0/0€€€€€€€€€∞!
К
0/1€€€€€€€€€
Ъ Ш
B__inference_model_layer_call_and_return_conditional_losses_3339171—<)*3456>?\]fghiqrПРЩЪЫЬ§•¬√ћЌќѕ„ЎхцЗИю€CҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p 

 
™ "LҐI
BЪ?
К
0/0€€€€€€€€€∞!
К
0/1€€€€€€€€€
Ъ Ш
B__inference_model_layer_call_and_return_conditional_losses_3339497—<)*3456>?\]fghiqrПРЩЪЫЬ§•¬√ћЌќѕ„ЎхцЗИю€CҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p

 
™ "LҐI
BЪ?
К
0/0€€€€€€€€€∞!
К
0/1€€€€€€€€€
Ъ р
'__inference_model_layer_call_fn_3337932ƒ<)*3456>?\]fghiqrПРЩЪЫЬ§•¬√ћЌќѕ„ЎхцЗИю€DҐA
:Ґ7
-К*
input_1€€€€€€€€€
p 

 
™ ">Ъ;
К
0€€€€€€€€€∞!
К
1€€€€€€€€€р
'__inference_model_layer_call_fn_3338457ƒ<)*3456>?\]fghiqrПРЩЪЫЬ§•¬√ћЌќѕ„ЎхцЗИю€DҐA
:Ґ7
-К*
input_1€€€€€€€€€
p

 
™ ">Ъ;
К
0€€€€€€€€€∞!
К
1€€€€€€€€€п
'__inference_model_layer_call_fn_3338762√<)*3456>?\]fghiqrПРЩЪЫЬ§•¬√ћЌќѕ„ЎхцЗИю€CҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p 

 
™ ">Ъ;
К
0€€€€€€€€€∞!
К
1€€€€€€€€€п
'__inference_model_layer_call_fn_3338845√<)*3456>?\]fghiqrПРЩЪЫЬ§•¬√ћЌќѕ„ЎхцЗИю€CҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p

 
™ ">Ъ;
К
0€€€€€€€€€∞!
К
1€€€€€€€€€Ґ
>__inference_p_layer_call_and_return_conditional_losses_3340334`ю€0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€∞!
Ъ z
#__inference_p_layer_call_fn_3340323Sю€0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€∞!ы
%__inference_signature_wrapper_3339582—<)*3456>?\]fghiqrПРЩЪЫЬ§•¬√ћЌќѕ„ЎхцЗИю€GҐD
Ґ 
=™:
8
input_1-К*
input_1€€€€€€€€€"H™E
!
pК
p€€€€€€€€€∞!
 
vК
v€€€€€€€€€°
>__inference_v_layer_call_and_return_conditional_losses_3340354_ЗИ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€
Ъ y
#__inference_v_layer_call_fn_3340343RЗИ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€