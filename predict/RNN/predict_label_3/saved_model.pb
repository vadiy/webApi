??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle???element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements(
handle???element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.8.02v2.8.0-0-g3f878cff5b68??
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:d*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
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
?
%simple_rnn_2/simple_rnn_cell_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*6
shared_name'%simple_rnn_2/simple_rnn_cell_2/kernel
?
9simple_rnn_2/simple_rnn_cell_2/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_2/simple_rnn_cell_2/kernel*
_output_shapes

:P*
dtype0
?
/simple_rnn_2/simple_rnn_cell_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*@
shared_name1/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel
?
Csimple_rnn_2/simple_rnn_cell_2/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel*
_output_shapes

:PP*
dtype0
?
#simple_rnn_2/simple_rnn_cell_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*4
shared_name%#simple_rnn_2/simple_rnn_cell_2/bias
?
7simple_rnn_2/simple_rnn_cell_2/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_2/simple_rnn_cell_2/bias*
_output_shapes
:P*
dtype0
?
%simple_rnn_3/simple_rnn_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*6
shared_name'%simple_rnn_3/simple_rnn_cell_3/kernel
?
9simple_rnn_3/simple_rnn_cell_3/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_3/simple_rnn_cell_3/kernel*
_output_shapes

:Pd*
dtype0
?
/simple_rnn_3/simple_rnn_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*@
shared_name1/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel
?
Csimple_rnn_3/simple_rnn_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel*
_output_shapes

:dd*
dtype0
?
#simple_rnn_3/simple_rnn_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#simple_rnn_3/simple_rnn_cell_3/bias
?
7simple_rnn_3/simple_rnn_cell_3/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_3/simple_rnn_cell_3/bias*
_output_shapes
:d*
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
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*=
shared_name.,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/m
?
@Adam/simple_rnn_2/simple_rnn_cell_2/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/m*
_output_shapes

:P*
dtype0
?
6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*G
shared_name86Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/m
?
JAdam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/m*
_output_shapes

:PP*
dtype0
?
*Adam/simple_rnn_2/simple_rnn_cell_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*;
shared_name,*Adam/simple_rnn_2/simple_rnn_cell_2/bias/m
?
>Adam/simple_rnn_2/simple_rnn_cell_2/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_2/simple_rnn_cell_2/bias/m*
_output_shapes
:P*
dtype0
?
,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*=
shared_name.,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/m
?
@Adam/simple_rnn_3/simple_rnn_cell_3/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/m*
_output_shapes

:Pd*
dtype0
?
6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*G
shared_name86Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/m
?
JAdam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/m*
_output_shapes

:dd*
dtype0
?
*Adam/simple_rnn_3/simple_rnn_cell_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*;
shared_name,*Adam/simple_rnn_3/simple_rnn_cell_3/bias/m
?
>Adam/simple_rnn_3/simple_rnn_cell_3/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_3/simple_rnn_cell_3/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
?
,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*=
shared_name.,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/v
?
@Adam/simple_rnn_2/simple_rnn_cell_2/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/v*
_output_shapes

:P*
dtype0
?
6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*G
shared_name86Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/v
?
JAdam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/v*
_output_shapes

:PP*
dtype0
?
*Adam/simple_rnn_2/simple_rnn_cell_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*;
shared_name,*Adam/simple_rnn_2/simple_rnn_cell_2/bias/v
?
>Adam/simple_rnn_2/simple_rnn_cell_2/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_2/simple_rnn_cell_2/bias/v*
_output_shapes
:P*
dtype0
?
,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*=
shared_name.,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/v
?
@Adam/simple_rnn_3/simple_rnn_cell_3/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/v*
_output_shapes

:Pd*
dtype0
?
6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*G
shared_name86Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/v
?
JAdam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/v*
_output_shapes

:dd*
dtype0
?
*Adam/simple_rnn_3/simple_rnn_cell_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*;
shared_name,*Adam/simple_rnn_3/simple_rnn_cell_3/bias/v
?
>Adam/simple_rnn_3/simple_rnn_cell_3/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_3/simple_rnn_cell_3/bias/v*
_output_shapes
:d*
dtype0

NoOpNoOp
?F
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?F
value?FB?F B?F
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer

signatures
#_self_saveable_object_factories
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature*
?
cell

state_spec
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
* &call_and_return_all_conditional_losses* 
?
!cell
"
state_spec
##_self_saveable_object_factories
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses*
?
#*_self_saveable_object_factories
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses* 
?

2kernel
3bias
#4_self_saveable_object_factories
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses*
?
;iter

<beta_1

=beta_2
	>decay
?learning_rate2m?3m?Am?Bm?Cm?Dm?Em?Fm?2v?3v?Av?Bv?Cv?Dv?Ev?Fv?*

@serving_default* 
* 
<
A0
B1
C2
D3
E4
F5
26
37*
<
A0
B1
C2
D3
E4
F5
26
37*
* 
?
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
?

Akernel
Brecurrent_kernel
Cbias
#L_self_saveable_object_factories
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q_random_generator
R__call__
*S&call_and_return_all_conditional_losses*
* 
* 

A0
B1
C2*

A0
B1
C2*
* 
?

Tstates
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses* 
* 
* 
* 
?

Dkernel
Erecurrent_kernel
Fbias
#__self_saveable_object_factories
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d_random_generator
e__call__
*f&call_and_return_all_conditional_losses*
* 
* 

D0
E1
F2*

D0
E1
F2*
* 
?

gstates
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

20
31*

20
31*
* 
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*
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
e_
VARIABLE_VALUE%simple_rnn_2/simple_rnn_cell_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_2/simple_rnn_cell_2/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_3/simple_rnn_cell_3/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_3/simple_rnn_cell_3/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

w0*
* 
* 
* 

A0
B1
C2*

A0
B1
C2*
* 
?
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 

D0
E1
F2*

D0
E1
F2*
* 
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

!0*
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

?total

?count
?	variables
?	keras_api*
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
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
?{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/simple_rnn_2/simple_rnn_cell_2/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/simple_rnn_3/simple_rnn_cell_3/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/simple_rnn_2/simple_rnn_cell_2/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/simple_rnn_3/simple_rnn_cell_3/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
"serving_default_simple_rnn_2_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall"serving_default_simple_rnn_2_input%simple_rnn_2/simple_rnn_cell_2/kernel#simple_rnn_2/simple_rnn_cell_2/bias/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel%simple_rnn_3/simple_rnn_cell_3/kernel#simple_rnn_3/simple_rnn_cell_3/bias/simple_rnn_3/simple_rnn_cell_3/recurrent_kerneldense_1/kerneldense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_10853102
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9simple_rnn_2/simple_rnn_cell_2/kernel/Read/ReadVariableOpCsimple_rnn_2/simple_rnn_cell_2/recurrent_kernel/Read/ReadVariableOp7simple_rnn_2/simple_rnn_cell_2/bias/Read/ReadVariableOp9simple_rnn_3/simple_rnn_cell_3/kernel/Read/ReadVariableOpCsimple_rnn_3/simple_rnn_cell_3/recurrent_kernel/Read/ReadVariableOp7simple_rnn_3/simple_rnn_cell_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp@Adam/simple_rnn_2/simple_rnn_cell_2/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_2/simple_rnn_cell_2/bias/m/Read/ReadVariableOp@Adam/simple_rnn_3/simple_rnn_cell_3/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_3/simple_rnn_cell_3/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp@Adam/simple_rnn_2/simple_rnn_cell_2/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_2/simple_rnn_cell_2/bias/v/Read/ReadVariableOp@Adam/simple_rnn_3/simple_rnn_cell_3/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_3/simple_rnn_cell_3/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
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
GPU 2J 8? **
f%R#
!__inference__traced_save_10854367
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%simple_rnn_2/simple_rnn_cell_2/kernel/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel#simple_rnn_2/simple_rnn_cell_2/bias%simple_rnn_3/simple_rnn_cell_3/kernel/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel#simple_rnn_3/simple_rnn_cell_3/biastotalcountAdam/dense_1/kernel/mAdam/dense_1/bias/m,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/m6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/m*Adam/simple_rnn_2/simple_rnn_cell_2/bias/m,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/m6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/m*Adam/simple_rnn_3/simple_rnn_cell_3/bias/mAdam/dense_1/kernel/vAdam/dense_1/bias/v,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/v6Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/v*Adam/simple_rnn_2/simple_rnn_cell_2/bias/v,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/v6Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/v*Adam/simple_rnn_3/simple_rnn_cell_3/bias/v*+
Tin$
"2 *
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_10854470??
?

f
G__inference_dropout_2_layer_call_and_return_conditional_losses_10852306

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????Ps
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Pm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?4
?
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10851827

inputs.
simple_rnn_cell_163_10851752:Pd*
simple_rnn_cell_163_10851754:d.
simple_rnn_cell_163_10851756:dd
identity??+simple_rnn_cell_163/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
+simple_rnn_cell_163/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_163_10851752simple_rnn_cell_163_10851754simple_rnn_cell_163_10851756*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10851712n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_163_10851752simple_rnn_cell_163_10851754simple_rnn_cell_163_10851756*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10851764*
condR
while_cond_10851763*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d|
NoOpNoOp,^simple_rnn_cell_163/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2Z
+simple_rnn_cell_163/StatefulPartitionedCall+simple_rnn_cell_163/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?
?
/__inference_simple_rnn_3_layer_call_fn_10853649

inputs
unknown:Pd
	unknown_0:d
	unknown_1:dd
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10852277o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?E
?
-sequential_1_simple_rnn_3_while_body_10851179P
Lsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_loop_counterV
Rsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_maximum_iterations/
+sequential_1_simple_rnn_3_while_placeholder1
-sequential_1_simple_rnn_3_while_placeholder_11
-sequential_1_simple_rnn_3_while_placeholder_2O
Ksequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_strided_slice_1_0?
?sequential_1_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0f
Tsequential_1_simple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resource_0:Pdc
Usequential_1_simple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resource_0:dh
Vsequential_1_simple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0:dd,
(sequential_1_simple_rnn_3_while_identity.
*sequential_1_simple_rnn_3_while_identity_1.
*sequential_1_simple_rnn_3_while_identity_2.
*sequential_1_simple_rnn_3_while_identity_3.
*sequential_1_simple_rnn_3_while_identity_4M
Isequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_strided_slice_1?
?sequential_1_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_3_tensorarrayunstack_tensorlistfromtensord
Rsequential_1_simple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resource:Pda
Ssequential_1_simple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resource:df
Tsequential_1_simple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resource:dd??Jsequential_1/simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOp?Isequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOp?Ksequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp?
Qsequential_1/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
Csequential_1/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_1_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0+sequential_1_simple_rnn_3_while_placeholderZsequential_1/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
Isequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOpTsequential_1_simple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
:sequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMulMatMulJsequential_1/simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem:item:0Qsequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Jsequential_1/simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOpUsequential_1_simple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
;sequential_1/simple_rnn_3/while/simple_rnn_cell_163/BiasAddBiasAddDsequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul:product:0Rsequential_1/simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Ksequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOpVsequential_1_simple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
<sequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul_1MatMul-sequential_1_simple_rnn_3_while_placeholder_2Ssequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
7sequential_1/simple_rnn_3/while/simple_rnn_cell_163/addAddV2Dsequential_1/simple_rnn_3/while/simple_rnn_cell_163/BiasAdd:output:0Fsequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
8sequential_1/simple_rnn_3/while/simple_rnn_cell_163/TanhTanh;sequential_1/simple_rnn_3/while/simple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????d?
Dsequential_1/simple_rnn_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_1_simple_rnn_3_while_placeholder_1+sequential_1_simple_rnn_3_while_placeholder<sequential_1/simple_rnn_3/while/simple_rnn_cell_163/Tanh:y:0*
_output_shapes
: *
element_dtype0:???g
%sequential_1/simple_rnn_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
#sequential_1/simple_rnn_3/while/addAddV2+sequential_1_simple_rnn_3_while_placeholder.sequential_1/simple_rnn_3/while/add/y:output:0*
T0*
_output_shapes
: i
'sequential_1/simple_rnn_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
%sequential_1/simple_rnn_3/while/add_1AddV2Lsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_loop_counter0sequential_1/simple_rnn_3/while/add_1/y:output:0*
T0*
_output_shapes
: ?
(sequential_1/simple_rnn_3/while/IdentityIdentity)sequential_1/simple_rnn_3/while/add_1:z:0%^sequential_1/simple_rnn_3/while/NoOp*
T0*
_output_shapes
: ?
*sequential_1/simple_rnn_3/while/Identity_1IdentityRsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_maximum_iterations%^sequential_1/simple_rnn_3/while/NoOp*
T0*
_output_shapes
: ?
*sequential_1/simple_rnn_3/while/Identity_2Identity'sequential_1/simple_rnn_3/while/add:z:0%^sequential_1/simple_rnn_3/while/NoOp*
T0*
_output_shapes
: ?
*sequential_1/simple_rnn_3/while/Identity_3IdentityTsequential_1/simple_rnn_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^sequential_1/simple_rnn_3/while/NoOp*
T0*
_output_shapes
: :????
*sequential_1/simple_rnn_3/while/Identity_4Identity<sequential_1/simple_rnn_3/while/simple_rnn_cell_163/Tanh:y:0%^sequential_1/simple_rnn_3/while/NoOp*
T0*'
_output_shapes
:?????????d?
$sequential_1/simple_rnn_3/while/NoOpNoOpK^sequential_1/simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOpJ^sequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOpL^sequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_1_simple_rnn_3_while_identity1sequential_1/simple_rnn_3/while/Identity:output:0"a
*sequential_1_simple_rnn_3_while_identity_13sequential_1/simple_rnn_3/while/Identity_1:output:0"a
*sequential_1_simple_rnn_3_while_identity_23sequential_1/simple_rnn_3/while/Identity_2:output:0"a
*sequential_1_simple_rnn_3_while_identity_33sequential_1/simple_rnn_3/while/Identity_3:output:0"a
*sequential_1_simple_rnn_3_while_identity_43sequential_1/simple_rnn_3/while/Identity_4:output:0"?
Isequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_strided_slice_1Ksequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_strided_slice_1_0"?
Ssequential_1_simple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resourceUsequential_1_simple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resource_0"?
Tsequential_1_simple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resourceVsequential_1_simple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0"?
Rsequential_1_simple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resourceTsequential_1_simple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resource_0"?
?sequential_1_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor?sequential_1_simple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2?
Jsequential_1/simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOpJsequential_1/simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOp2?
Isequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOpIsequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOp2?
Ksequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOpKsequential_1/simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?=
?
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10853578

inputsD
2simple_rnn_cell_162_matmul_readvariableop_resource:PA
3simple_rnn_cell_162_biasadd_readvariableop_resource:PF
4simple_rnn_cell_162_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_162/BiasAdd/ReadVariableOp?)simple_rnn_cell_162/MatMul/ReadVariableOp?+simple_rnn_cell_162/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
)simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_162_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_162/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_162_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_162/BiasAddBiasAdd$simple_rnn_cell_162/MatMul:product:02simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_162_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_162/MatMul_1MatMulzeros:output:03simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_162/addAddV2$simple_rnn_cell_162/BiasAdd:output:0&simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_162/TanhTanhsimple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????Pn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_162_matmul_readvariableop_resource3simple_rnn_cell_162_biasadd_readvariableop_resource4simple_rnn_cell_162_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10853512*
condR
while_cond_10853511*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_162/BiasAdd/ReadVariableOp*^simple_rnn_cell_162/MatMul/ReadVariableOp,^simple_rnn_cell_162/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_162/BiasAdd/ReadVariableOp*simple_rnn_cell_162/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_162/MatMul/ReadVariableOp)simple_rnn_cell_162/MatMul/ReadVariableOp2Z
+simple_rnn_cell_162/MatMul_1/ReadVariableOp+simple_rnn_cell_162/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_10853511
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10853511___redundant_placeholder06
2while_while_cond_10853511___redundant_placeholder16
2while_while_cond_10853511___redundant_placeholder26
2while_while_cond_10853511___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?
?
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10854234

inputs
states_00
matmul_readvariableop_resource:Pd-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????dG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????dY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????P:?????????d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0
?
?
while_cond_10854014
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10854014___redundant_placeholder06
2while_while_cond_10854014___redundant_placeholder16
2while_while_cond_10854014___redundant_placeholder26
2while_while_cond_10854014___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?-
?
while_body_10852364
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_162_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_162_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_162_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_162_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_162/MatMul/ReadVariableOp?1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_162_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_162/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_162/BiasAddBiasAdd*while/simple_rnn_cell_162/MatMul:product:08while/simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_162/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_162/addAddV2*while/simple_rnn_cell_162/BiasAdd:output:0,while/simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_162/TanhTanh!while/simple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_162/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_162/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_162/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_162/MatMul/ReadVariableOp2^while/simple_rnn_cell_162/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_162_biasadd_readvariableop_resource;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_162_matmul_1_readvariableop_resource<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_162_matmul_readvariableop_resource:while_simple_rnn_cell_162_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_162/MatMul/ReadVariableOp/while/simple_rnn_cell_162/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?	
?
/__inference_sequential_1_layer_call_fn_10852527
simple_rnn_2_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852487o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_2_input
?=
?
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10852072

inputsD
2simple_rnn_cell_163_matmul_readvariableop_resource:PdA
3simple_rnn_cell_163_biasadd_readvariableop_resource:dF
4simple_rnn_cell_163_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_163/BiasAdd/ReadVariableOp?)simple_rnn_cell_163/MatMul/ReadVariableOp?+simple_rnn_cell_163/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
)simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_163_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_163/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_163_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_163/BiasAddBiasAdd$simple_rnn_cell_163/MatMul:product:02simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_163_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_163/MatMul_1MatMulzeros:output:03simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_163/addAddV2$simple_rnn_cell_163/BiasAdd:output:0&simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_163/TanhTanhsimple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_163_matmul_readvariableop_resource3simple_rnn_cell_163_biasadd_readvariableop_resource4simple_rnn_cell_163_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10852006*
condR
while_cond_10852005*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_163/BiasAdd/ReadVariableOp*^simple_rnn_cell_163/MatMul/ReadVariableOp,^simple_rnn_cell_163/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_163/BiasAdd/ReadVariableOp*simple_rnn_cell_163/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_163/MatMul/ReadVariableOp)simple_rnn_cell_163/MatMul/ReadVariableOp2Z
+simple_rnn_cell_163/MatMul_1/ReadVariableOp+simple_rnn_cell_163/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?	
?
/__inference_sequential_1_layer_call_fn_10852123
simple_rnn_2_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852104o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_2_input
?
?
/__inference_simple_rnn_2_layer_call_fn_10853135

inputs
unknown:P
	unknown_0:P
	unknown_1:PP
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10851950s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852552
simple_rnn_2_input'
simple_rnn_2_10852530:P#
simple_rnn_2_10852532:P'
simple_rnn_2_10852534:PP'
simple_rnn_3_10852538:Pd#
simple_rnn_3_10852540:d'
simple_rnn_3_10852542:dd"
dense_1_10852546:d
dense_1_10852548:
identity??dense_1/StatefulPartitionedCall?$simple_rnn_2/StatefulPartitionedCall?$simple_rnn_3/StatefulPartitionedCall?
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputsimple_rnn_2_10852530simple_rnn_2_10852532simple_rnn_2_10852534*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10851950?
dropout_2/PartitionedCallPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_10851963?
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0simple_rnn_3_10852538simple_rnn_3_10852540simple_rnn_3_10852542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10852072?
dropout_3/PartitionedCallPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_10852085?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_1_10852546dense_1_10852548*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_10852097w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_2_input
?=
?
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10853973

inputsD
2simple_rnn_cell_163_matmul_readvariableop_resource:PdA
3simple_rnn_cell_163_biasadd_readvariableop_resource:dF
4simple_rnn_cell_163_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_163/BiasAdd/ReadVariableOp?)simple_rnn_cell_163/MatMul/ReadVariableOp?+simple_rnn_cell_163/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
)simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_163_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_163/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_163_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_163/BiasAddBiasAdd$simple_rnn_cell_163/MatMul:product:02simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_163_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_163/MatMul_1MatMulzeros:output:03simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_163/addAddV2$simple_rnn_cell_163/BiasAdd:output:0&simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_163/TanhTanhsimple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_163_matmul_readvariableop_resource3simple_rnn_cell_163_biasadd_readvariableop_resource4simple_rnn_cell_163_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10853907*
condR
while_cond_10853906*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_163/BiasAdd/ReadVariableOp*^simple_rnn_cell_163/MatMul/ReadVariableOp,^simple_rnn_cell_163/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_163/BiasAdd/ReadVariableOp*simple_rnn_cell_163/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_163/MatMul/ReadVariableOp)simple_rnn_cell_163/MatMul/ReadVariableOp2Z
+simple_rnn_cell_163/MatMul_1/ReadVariableOp+simple_rnn_cell_163/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?E
?
-sequential_1_simple_rnn_2_while_body_10851074P
Lsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_loop_counterV
Rsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_maximum_iterations/
+sequential_1_simple_rnn_2_while_placeholder1
-sequential_1_simple_rnn_2_while_placeholder_11
-sequential_1_simple_rnn_2_while_placeholder_2O
Ksequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_strided_slice_1_0?
?sequential_1_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0f
Tsequential_1_simple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resource_0:Pc
Usequential_1_simple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resource_0:Ph
Vsequential_1_simple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0:PP,
(sequential_1_simple_rnn_2_while_identity.
*sequential_1_simple_rnn_2_while_identity_1.
*sequential_1_simple_rnn_2_while_identity_2.
*sequential_1_simple_rnn_2_while_identity_3.
*sequential_1_simple_rnn_2_while_identity_4M
Isequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_strided_slice_1?
?sequential_1_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_2_tensorarrayunstack_tensorlistfromtensord
Rsequential_1_simple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resource:Pa
Ssequential_1_simple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resource:Pf
Tsequential_1_simple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resource:PP??Jsequential_1/simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOp?Isequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOp?Ksequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp?
Qsequential_1/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Csequential_1/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_1_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0+sequential_1_simple_rnn_2_while_placeholderZsequential_1/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
Isequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOpTsequential_1_simple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
:sequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMulMatMulJsequential_1/simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Qsequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Jsequential_1/simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOpUsequential_1_simple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
;sequential_1/simple_rnn_2/while/simple_rnn_cell_162/BiasAddBiasAddDsequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul:product:0Rsequential_1/simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Ksequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOpVsequential_1_simple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
<sequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul_1MatMul-sequential_1_simple_rnn_2_while_placeholder_2Ssequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
7sequential_1/simple_rnn_2/while/simple_rnn_cell_162/addAddV2Dsequential_1/simple_rnn_2/while/simple_rnn_cell_162/BiasAdd:output:0Fsequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
8sequential_1/simple_rnn_2/while/simple_rnn_cell_162/TanhTanh;sequential_1/simple_rnn_2/while/simple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????P?
Dsequential_1/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_1_simple_rnn_2_while_placeholder_1+sequential_1_simple_rnn_2_while_placeholder<sequential_1/simple_rnn_2/while/simple_rnn_cell_162/Tanh:y:0*
_output_shapes
: *
element_dtype0:???g
%sequential_1/simple_rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
#sequential_1/simple_rnn_2/while/addAddV2+sequential_1_simple_rnn_2_while_placeholder.sequential_1/simple_rnn_2/while/add/y:output:0*
T0*
_output_shapes
: i
'sequential_1/simple_rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
%sequential_1/simple_rnn_2/while/add_1AddV2Lsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_loop_counter0sequential_1/simple_rnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: ?
(sequential_1/simple_rnn_2/while/IdentityIdentity)sequential_1/simple_rnn_2/while/add_1:z:0%^sequential_1/simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ?
*sequential_1/simple_rnn_2/while/Identity_1IdentityRsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_maximum_iterations%^sequential_1/simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ?
*sequential_1/simple_rnn_2/while/Identity_2Identity'sequential_1/simple_rnn_2/while/add:z:0%^sequential_1/simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ?
*sequential_1/simple_rnn_2/while/Identity_3IdentityTsequential_1/simple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^sequential_1/simple_rnn_2/while/NoOp*
T0*
_output_shapes
: :????
*sequential_1/simple_rnn_2/while/Identity_4Identity<sequential_1/simple_rnn_2/while/simple_rnn_cell_162/Tanh:y:0%^sequential_1/simple_rnn_2/while/NoOp*
T0*'
_output_shapes
:?????????P?
$sequential_1/simple_rnn_2/while/NoOpNoOpK^sequential_1/simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOpJ^sequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOpL^sequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_1_simple_rnn_2_while_identity1sequential_1/simple_rnn_2/while/Identity:output:0"a
*sequential_1_simple_rnn_2_while_identity_13sequential_1/simple_rnn_2/while/Identity_1:output:0"a
*sequential_1_simple_rnn_2_while_identity_23sequential_1/simple_rnn_2/while/Identity_2:output:0"a
*sequential_1_simple_rnn_2_while_identity_33sequential_1/simple_rnn_2/while/Identity_3:output:0"a
*sequential_1_simple_rnn_2_while_identity_43sequential_1/simple_rnn_2/while/Identity_4:output:0"?
Isequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_strided_slice_1Ksequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_strided_slice_1_0"?
Ssequential_1_simple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resourceUsequential_1_simple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resource_0"?
Tsequential_1_simple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resourceVsequential_1_simple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0"?
Rsequential_1_simple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resourceTsequential_1_simple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resource_0"?
?sequential_1_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor?sequential_1_simple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2?
Jsequential_1/simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOpJsequential_1/simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOp2?
Isequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOpIsequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOp2?
Ksequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOpKsequential_1/simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?
e
G__inference_dropout_2_layer_call_and_return_conditional_losses_10851963

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?=
?
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10852430

inputsD
2simple_rnn_cell_162_matmul_readvariableop_resource:PA
3simple_rnn_cell_162_biasadd_readvariableop_resource:PF
4simple_rnn_cell_162_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_162/BiasAdd/ReadVariableOp?)simple_rnn_cell_162/MatMul/ReadVariableOp?+simple_rnn_cell_162/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
)simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_162_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_162/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_162_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_162/BiasAddBiasAdd$simple_rnn_cell_162/MatMul:product:02simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_162_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_162/MatMul_1MatMulzeros:output:03simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_162/addAddV2$simple_rnn_cell_162/BiasAdd:output:0&simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_162/TanhTanhsimple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????Pn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_162_matmul_readvariableop_resource3simple_rnn_cell_162_biasadd_readvariableop_resource4simple_rnn_cell_162_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10852364*
condR
while_cond_10852363*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_162/BiasAdd/ReadVariableOp*^simple_rnn_cell_162/MatMul/ReadVariableOp,^simple_rnn_cell_162/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_162/BiasAdd/ReadVariableOp*simple_rnn_cell_162/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_162/MatMul/ReadVariableOp)simple_rnn_cell_162/MatMul/ReadVariableOp2Z
+simple_rnn_cell_162/MatMul_1/ReadVariableOp+simple_rnn_cell_162/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_10852005
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10852005___redundant_placeholder06
2while_while_cond_10852005___redundant_placeholder16
2while_while_cond_10852005___redundant_placeholder26
2while_while_cond_10852005___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
-sequential_1_simple_rnn_3_while_cond_10851178P
Lsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_loop_counterV
Rsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_maximum_iterations/
+sequential_1_simple_rnn_3_while_placeholder1
-sequential_1_simple_rnn_3_while_placeholder_11
-sequential_1_simple_rnn_3_while_placeholder_2R
Nsequential_1_simple_rnn_3_while_less_sequential_1_simple_rnn_3_strided_slice_1j
fsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_cond_10851178___redundant_placeholder0j
fsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_cond_10851178___redundant_placeholder1j
fsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_cond_10851178___redundant_placeholder2j
fsequential_1_simple_rnn_3_while_sequential_1_simple_rnn_3_while_cond_10851178___redundant_placeholder3,
(sequential_1_simple_rnn_3_while_identity
?
$sequential_1/simple_rnn_3/while/LessLess+sequential_1_simple_rnn_3_while_placeholderNsequential_1_simple_rnn_3_while_less_sequential_1_simple_rnn_3_strided_slice_1*
T0*
_output_shapes
: 
(sequential_1/simple_rnn_3/while/IdentityIdentity(sequential_1/simple_rnn_3/while/Less:z:0*
T0
*
_output_shapes
: "]
(sequential_1_simple_rnn_3_while_identity1sequential_1/simple_rnn_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
e
G__inference_dropout_2_layer_call_and_return_conditional_losses_10853593

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?9
?
 simple_rnn_2_while_body_108528876
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_25
1simple_rnn_2_while_simple_rnn_2_strided_slice_1_0q
msimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resource_0:PV
Hsimple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resource_0:P[
Isimple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0:PP
simple_rnn_2_while_identity!
simple_rnn_2_while_identity_1!
simple_rnn_2_while_identity_2!
simple_rnn_2_while_identity_3!
simple_rnn_2_while_identity_43
/simple_rnn_2_while_simple_rnn_2_strided_slice_1o
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resource:PT
Fsimple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resource:PY
Gsimple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resource:PP??=simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOp?<simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOp?>simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp?
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_2_while_placeholderMsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
<simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
-simple_rnn_2/while/simple_rnn_cell_162/MatMulMatMul=simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
=simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
.simple_rnn_2/while/simple_rnn_cell_162/BiasAddBiasAdd7simple_rnn_2/while/simple_rnn_cell_162/MatMul:product:0Esimple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
>simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
/simple_rnn_2/while/simple_rnn_cell_162/MatMul_1MatMul simple_rnn_2_while_placeholder_2Fsimple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_2/while/simple_rnn_cell_162/addAddV27simple_rnn_2/while/simple_rnn_cell_162/BiasAdd:output:09simple_rnn_2/while/simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_2/while/simple_rnn_cell_162/TanhTanh.simple_rnn_2/while/simple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????P?
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_2_while_placeholder_1simple_rnn_2_while_placeholder/simple_rnn_2/while/simple_rnn_cell_162/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_2/while/addAddV2simple_rnn_2_while_placeholder!simple_rnn_2/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_2/while/add_1AddV22simple_rnn_2_while_simple_rnn_2_while_loop_counter#simple_rnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/add_1:z:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_2/while/Identity_1Identity8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_2/while/Identity_2Identitysimple_rnn_2/while/add:z:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_2/while/Identity_3IdentityGsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_2/while/Identity_4Identity/simple_rnn_2/while/simple_rnn_cell_162/Tanh:y:0^simple_rnn_2/while/NoOp*
T0*'
_output_shapes
:?????????P?
simple_rnn_2/while/NoOpNoOp>^simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOp?^simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0"G
simple_rnn_2_while_identity_1&simple_rnn_2/while/Identity_1:output:0"G
simple_rnn_2_while_identity_2&simple_rnn_2/while/Identity_2:output:0"G
simple_rnn_2_while_identity_3&simple_rnn_2/while/Identity_3:output:0"G
simple_rnn_2_while_identity_4&simple_rnn_2/while/Identity_4:output:0"d
/simple_rnn_2_while_simple_rnn_2_strided_slice_11simple_rnn_2_while_simple_rnn_2_strided_slice_1_0"?
Fsimple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resourceHsimple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resource_0"?
Gsimple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resourceIsimple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0"?
Esimple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resourceGsimple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resource_0"?
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensormsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2~
=simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOp=simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOp2|
<simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOp<simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOp2?
>simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp>simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?-
?
while_body_10853907
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_163_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_163_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_163_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_163_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_163/MatMul/ReadVariableOp?1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_163_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_163/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_163/BiasAddBiasAdd*while/simple_rnn_cell_163/MatMul:product:08while/simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_163/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_163/addAddV2*while/simple_rnn_cell_163/BiasAdd:output:0,while/simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_163/TanhTanh!while/simple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_163/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_163/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_163/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_163/MatMul/ReadVariableOp2^while/simple_rnn_cell_163/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_163_biasadd_readvariableop_resource;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_163_matmul_1_readvariableop_resource<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_163_matmul_readvariableop_resource:while_simple_rnn_cell_163_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_163/MatMul/ReadVariableOp/while/simple_rnn_cell_163/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10854251

inputs
states_00
matmul_readvariableop_resource:Pd-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????dG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????dY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????P:?????????d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0
?-
?
while_body_10853296
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_162_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_162_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_162_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_162_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_162/MatMul/ReadVariableOp?1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_162_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_162/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_162/BiasAddBiasAdd*while/simple_rnn_cell_162/MatMul:product:08while/simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_162/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_162/addAddV2*while/simple_rnn_cell_162/BiasAdd:output:0,while/simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_162/TanhTanh!while/simple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_162/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_162/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_162/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_162/MatMul/ReadVariableOp2^while/simple_rnn_cell_162/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_162_biasadd_readvariableop_resource;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_162_matmul_1_readvariableop_resource<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_162_matmul_readvariableop_resource:while_simple_rnn_cell_162_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_162/MatMul/ReadVariableOp/while/simple_rnn_cell_162/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_10852085

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
 simple_rnn_3_while_cond_108527716
2simple_rnn_3_while_simple_rnn_3_while_loop_counter<
8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations"
simple_rnn_3_while_placeholder$
 simple_rnn_3_while_placeholder_1$
 simple_rnn_3_while_placeholder_28
4simple_rnn_3_while_less_simple_rnn_3_strided_slice_1P
Lsimple_rnn_3_while_simple_rnn_3_while_cond_10852771___redundant_placeholder0P
Lsimple_rnn_3_while_simple_rnn_3_while_cond_10852771___redundant_placeholder1P
Lsimple_rnn_3_while_simple_rnn_3_while_cond_10852771___redundant_placeholder2P
Lsimple_rnn_3_while_simple_rnn_3_while_cond_10852771___redundant_placeholder3
simple_rnn_3_while_identity
?
simple_rnn_3/while/LessLesssimple_rnn_3_while_placeholder4simple_rnn_3_while_less_simple_rnn_3_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_3_while_identity$simple_rnn_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?

?
 simple_rnn_2_while_cond_108528866
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_28
4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1P
Lsimple_rnn_2_while_simple_rnn_2_while_cond_10852886___redundant_placeholder0P
Lsimple_rnn_2_while_simple_rnn_2_while_cond_10852886___redundant_placeholder1P
Lsimple_rnn_2_while_simple_rnn_2_while_cond_10852886___redundant_placeholder2P
Lsimple_rnn_2_while_simple_rnn_2_while_cond_10852886___redundant_placeholder3
simple_rnn_2_while_identity
?
simple_rnn_2/while/LessLesssimple_rnn_2_while_placeholder4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?
H
,__inference_dropout_3_layer_call_fn_10854086

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_10852085`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?=
?
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10853470

inputsD
2simple_rnn_cell_162_matmul_readvariableop_resource:PA
3simple_rnn_cell_162_biasadd_readvariableop_resource:PF
4simple_rnn_cell_162_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_162/BiasAdd/ReadVariableOp?)simple_rnn_cell_162/MatMul/ReadVariableOp?+simple_rnn_cell_162/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
)simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_162_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_162/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_162_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_162/BiasAddBiasAdd$simple_rnn_cell_162/MatMul:product:02simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_162_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_162/MatMul_1MatMulzeros:output:03simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_162/addAddV2$simple_rnn_cell_162/BiasAdd:output:0&simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_162/TanhTanhsimple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????Pn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_162_matmul_readvariableop_resource3simple_rnn_cell_162_biasadd_readvariableop_resource4simple_rnn_cell_162_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10853404*
condR
while_cond_10853403*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_162/BiasAdd/ReadVariableOp*^simple_rnn_cell_162/MatMul/ReadVariableOp,^simple_rnn_cell_162/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_162/BiasAdd/ReadVariableOp*simple_rnn_cell_162/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_162/MatMul/ReadVariableOp)simple_rnn_cell_162/MatMul/ReadVariableOp2Z
+simple_rnn_cell_162/MatMul_1/ReadVariableOp+simple_rnn_cell_162/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_10854096

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
E__inference_dense_1_layer_call_and_return_conditional_losses_10854127

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
f
G__inference_dropout_3_layer_call_and_return_conditional_losses_10854108

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????dC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????do
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????di
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????dY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
H
,__inference_dropout_2_layer_call_fn_10853583

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_10851963d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
while_cond_10851763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10851763___redundant_placeholder06
2while_while_cond_10851763___redundant_placeholder16
2while_while_cond_10851763___redundant_placeholder26
2while_while_cond_10851763___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_10853690
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10853690___redundant_placeholder06
2while_while_cond_10853690___redundant_placeholder16
2while_while_cond_10853690___redundant_placeholder26
2while_while_cond_10853690___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?>
?
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10853362
inputs_0D
2simple_rnn_cell_162_matmul_readvariableop_resource:PA
3simple_rnn_cell_162_biasadd_readvariableop_resource:PF
4simple_rnn_cell_162_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_162/BiasAdd/ReadVariableOp?)simple_rnn_cell_162/MatMul/ReadVariableOp?+simple_rnn_cell_162/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
)simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_162_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_162/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_162_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_162/BiasAddBiasAdd$simple_rnn_cell_162/MatMul:product:02simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_162_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_162/MatMul_1MatMulzeros:output:03simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_162/addAddV2$simple_rnn_cell_162/BiasAdd:output:0&simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_162/TanhTanhsimple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????Pn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_162_matmul_readvariableop_resource3simple_rnn_cell_162_biasadd_readvariableop_resource4simple_rnn_cell_162_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10853296*
condR
while_cond_10853295*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????Pk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????P?
NoOpNoOp+^simple_rnn_cell_162/BiasAdd/ReadVariableOp*^simple_rnn_cell_162/MatMul/ReadVariableOp,^simple_rnn_cell_162/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2X
*simple_rnn_cell_162/BiasAdd/ReadVariableOp*simple_rnn_cell_162/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_162/MatMul/ReadVariableOp)simple_rnn_cell_162/MatMul/ReadVariableOp2Z
+simple_rnn_cell_162/MatMul_1/ReadVariableOp+simple_rnn_cell_162/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?4
?
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10851668

inputs.
simple_rnn_cell_163_10851593:Pd*
simple_rnn_cell_163_10851595:d.
simple_rnn_cell_163_10851597:dd
identity??+simple_rnn_cell_163/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
+simple_rnn_cell_163/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_163_10851593simple_rnn_cell_163_10851595simple_rnn_cell_163_10851597*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10851592n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_163_10851593simple_rnn_cell_163_10851595simple_rnn_cell_163_10851597*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10851605*
condR
while_cond_10851604*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d|
NoOpNoOp,^simple_rnn_cell_163/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2Z
+simple_rnn_cell_163/StatefulPartitionedCall+simple_rnn_cell_163/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?!
?
while_body_10851605
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_163_10851627_0:Pd2
$while_simple_rnn_cell_163_10851629_0:d6
$while_simple_rnn_cell_163_10851631_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_163_10851627:Pd0
"while_simple_rnn_cell_163_10851629:d4
"while_simple_rnn_cell_163_10851631:dd??1while/simple_rnn_cell_163/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
1while/simple_rnn_cell_163/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_163_10851627_0$while_simple_rnn_cell_163_10851629_0$while_simple_rnn_cell_163_10851631_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10851592?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_163/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :????
while/Identity_4Identity:while/simple_rnn_cell_163/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp2^while/simple_rnn_cell_163/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_163_10851627$while_simple_rnn_cell_163_10851627_0"J
"while_simple_rnn_cell_163_10851629$while_simple_rnn_cell_163_10851629_0"J
"while_simple_rnn_cell_163_10851631$while_simple_rnn_cell_163_10851631_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2f
1while/simple_rnn_cell_163/StatefulPartitionedCall1while/simple_rnn_cell_163/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10854172

inputs
states_00
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:P2
 matmul_1_readvariableop_resource:PP
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Px
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????PG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????PW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????PY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????P?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????P: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????P
"
_user_specified_name
states/0
?
?
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10851300

inputs

states0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:P2
 matmul_1_readvariableop_resource:PP
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Px
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????PG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????PW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????PY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????P?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????P: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????P
 
_user_specified_namestates
?4
?
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10851535

inputs.
simple_rnn_cell_162_10851460:P*
simple_rnn_cell_162_10851462:P.
simple_rnn_cell_162_10851464:PP
identity??+simple_rnn_cell_162/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
+simple_rnn_cell_162/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_162_10851460simple_rnn_cell_162_10851462simple_rnn_cell_162_10851464*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????P:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10851420n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_162_10851460simple_rnn_cell_162_10851462simple_rnn_cell_162_10851464*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10851472*
condR
while_cond_10851471*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????Pk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????P|
NoOpNoOp,^simple_rnn_cell_162/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+simple_rnn_cell_162/StatefulPartitionedCall+simple_rnn_cell_162/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
e
,__inference_dropout_3_layer_call_fn_10854091

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_10852153o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?4
?
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10851376

inputs.
simple_rnn_cell_162_10851301:P*
simple_rnn_cell_162_10851303:P.
simple_rnn_cell_162_10851305:PP
identity??+simple_rnn_cell_162/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
+simple_rnn_cell_162/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_162_10851301simple_rnn_cell_162_10851303simple_rnn_cell_162_10851305*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????P:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10851300n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_162_10851301simple_rnn_cell_162_10851303simple_rnn_cell_162_10851305*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10851313*
condR
while_cond_10851312*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????Pk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????P|
NoOpNoOp,^simple_rnn_cell_162/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+simple_rnn_cell_162/StatefulPartitionedCall+simple_rnn_cell_162/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852487

inputs'
simple_rnn_2_10852465:P#
simple_rnn_2_10852467:P'
simple_rnn_2_10852469:PP'
simple_rnn_3_10852473:Pd#
simple_rnn_3_10852475:d'
simple_rnn_3_10852477:dd"
dense_1_10852481:d
dense_1_10852483:
identity??dense_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?$simple_rnn_2/StatefulPartitionedCall?$simple_rnn_3/StatefulPartitionedCall?
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_2_10852465simple_rnn_2_10852467simple_rnn_2_10852469*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10852430?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_10852306?
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0simple_rnn_3_10852473simple_rnn_3_10852475simple_rnn_3_10852477*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10852277?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_10852153?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_1_10852481dense_1_10852483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_10852097w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?
 simple_rnn_3_while_body_108527726
2simple_rnn_3_while_simple_rnn_3_while_loop_counter<
8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations"
simple_rnn_3_while_placeholder$
 simple_rnn_3_while_placeholder_1$
 simple_rnn_3_while_placeholder_25
1simple_rnn_3_while_simple_rnn_3_strided_slice_1_0q
msimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resource_0:PdV
Hsimple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resource_0:d[
Isimple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0:dd
simple_rnn_3_while_identity!
simple_rnn_3_while_identity_1!
simple_rnn_3_while_identity_2!
simple_rnn_3_while_identity_3!
simple_rnn_3_while_identity_43
/simple_rnn_3_while_simple_rnn_3_strided_slice_1o
ksimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resource:PdT
Fsimple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resource:dY
Gsimple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resource:dd??=simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOp?<simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOp?>simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp?
Dsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
6simple_rnn_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_3_while_placeholderMsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
<simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
-simple_rnn_3/while/simple_rnn_cell_163/MatMulMatMul=simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
=simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
.simple_rnn_3/while/simple_rnn_cell_163/BiasAddBiasAdd7simple_rnn_3/while/simple_rnn_cell_163/MatMul:product:0Esimple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
>simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
/simple_rnn_3/while/simple_rnn_cell_163/MatMul_1MatMul simple_rnn_3_while_placeholder_2Fsimple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_3/while/simple_rnn_cell_163/addAddV27simple_rnn_3/while/simple_rnn_cell_163/BiasAdd:output:09simple_rnn_3/while/simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_3/while/simple_rnn_cell_163/TanhTanh.simple_rnn_3/while/simple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????d?
7simple_rnn_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_3_while_placeholder_1simple_rnn_3_while_placeholder/simple_rnn_3/while/simple_rnn_cell_163/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_3/while/addAddV2simple_rnn_3_while_placeholder!simple_rnn_3/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_3/while/add_1AddV22simple_rnn_3_while_simple_rnn_3_while_loop_counter#simple_rnn_3/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while/add_1:z:0^simple_rnn_3/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_3/while/Identity_1Identity8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations^simple_rnn_3/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_3/while/Identity_2Identitysimple_rnn_3/while/add:z:0^simple_rnn_3/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_3/while/Identity_3IdentityGsimple_rnn_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_3/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_3/while/Identity_4Identity/simple_rnn_3/while/simple_rnn_cell_163/Tanh:y:0^simple_rnn_3/while/NoOp*
T0*'
_output_shapes
:?????????d?
simple_rnn_3/while/NoOpNoOp>^simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOp?^simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_3_while_identity$simple_rnn_3/while/Identity:output:0"G
simple_rnn_3_while_identity_1&simple_rnn_3/while/Identity_1:output:0"G
simple_rnn_3_while_identity_2&simple_rnn_3/while/Identity_2:output:0"G
simple_rnn_3_while_identity_3&simple_rnn_3/while/Identity_3:output:0"G
simple_rnn_3_while_identity_4&simple_rnn_3/while/Identity_4:output:0"d
/simple_rnn_3_while_simple_rnn_3_strided_slice_11simple_rnn_3_while_simple_rnn_3_strided_slice_1_0"?
Fsimple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resourceHsimple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resource_0"?
Gsimple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resourceIsimple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0"?
Esimple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resourceGsimple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resource_0"?
ksimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensormsimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2~
=simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOp=simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOp2|
<simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOp<simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOp2?
>simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp>simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
/__inference_simple_rnn_3_layer_call_fn_10853627
inputs_0
unknown:Pd
	unknown_0:d
	unknown_1:dd
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10851827o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?
?
while_cond_10853906
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10853906___redundant_placeholder06
2while_while_cond_10853906___redundant_placeholder16
2while_while_cond_10853906___redundant_placeholder26
2while_while_cond_10853906___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?

?
 simple_rnn_3_while_cond_108529986
2simple_rnn_3_while_simple_rnn_3_while_loop_counter<
8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations"
simple_rnn_3_while_placeholder$
 simple_rnn_3_while_placeholder_1$
 simple_rnn_3_while_placeholder_28
4simple_rnn_3_while_less_simple_rnn_3_strided_slice_1P
Lsimple_rnn_3_while_simple_rnn_3_while_cond_10852998___redundant_placeholder0P
Lsimple_rnn_3_while_simple_rnn_3_while_cond_10852998___redundant_placeholder1P
Lsimple_rnn_3_while_simple_rnn_3_while_cond_10852998___redundant_placeholder2P
Lsimple_rnn_3_while_simple_rnn_3_while_cond_10852998___redundant_placeholder3
simple_rnn_3_while_identity
?
simple_rnn_3/while/LessLesssimple_rnn_3_while_placeholder4simple_rnn_3_while_less_simple_rnn_3_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_3_while_identity$simple_rnn_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10851712

inputs

states0
matmul_readvariableop_resource:Pd-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????dG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????dY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????P:?????????d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates
?>
?
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10853254
inputs_0D
2simple_rnn_cell_162_matmul_readvariableop_resource:PA
3simple_rnn_cell_162_biasadd_readvariableop_resource:PF
4simple_rnn_cell_162_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_162/BiasAdd/ReadVariableOp?)simple_rnn_cell_162/MatMul/ReadVariableOp?+simple_rnn_cell_162/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
)simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_162_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_162/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_162_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_162/BiasAddBiasAdd$simple_rnn_cell_162/MatMul:product:02simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_162_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_162/MatMul_1MatMulzeros:output:03simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_162/addAddV2$simple_rnn_cell_162/BiasAdd:output:0&simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_162/TanhTanhsimple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????Pn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_162_matmul_readvariableop_resource3simple_rnn_cell_162_biasadd_readvariableop_resource4simple_rnn_cell_162_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10853188*
condR
while_cond_10853187*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????Pk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????P?
NoOpNoOp+^simple_rnn_cell_162/BiasAdd/ReadVariableOp*^simple_rnn_cell_162/MatMul/ReadVariableOp,^simple_rnn_cell_162/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2X
*simple_rnn_cell_162/BiasAdd/ReadVariableOp*simple_rnn_cell_162/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_162/MatMul/ReadVariableOp)simple_rnn_cell_162/MatMul/ReadVariableOp2Z
+simple_rnn_cell_162/MatMul_1/ReadVariableOp+simple_rnn_cell_162/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?

f
G__inference_dropout_2_layer_call_and_return_conditional_losses_10853605

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????Ps
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Pm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?	
?
E__inference_dense_1_layer_call_and_return_conditional_losses_10852097

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
while_cond_10851604
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10851604___redundant_placeholder06
2while_while_cond_10851604___redundant_placeholder16
2while_while_cond_10851604___redundant_placeholder26
2while_while_cond_10851604___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?	
?
/__inference_sequential_1_layer_call_fn_10852604

inputs
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852104o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?=
?
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10852277

inputsD
2simple_rnn_cell_163_matmul_readvariableop_resource:PdA
3simple_rnn_cell_163_biasadd_readvariableop_resource:dF
4simple_rnn_cell_163_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_163/BiasAdd/ReadVariableOp?)simple_rnn_cell_163/MatMul/ReadVariableOp?+simple_rnn_cell_163/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
)simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_163_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_163/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_163_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_163/BiasAddBiasAdd$simple_rnn_cell_163/MatMul:product:02simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_163_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_163/MatMul_1MatMulzeros:output:03simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_163/addAddV2$simple_rnn_cell_163/BiasAdd:output:0&simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_163/TanhTanhsimple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_163_matmul_readvariableop_resource3simple_rnn_cell_163_biasadd_readvariableop_resource4simple_rnn_cell_163_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10852211*
condR
while_cond_10852210*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_163/BiasAdd/ReadVariableOp*^simple_rnn_cell_163/MatMul/ReadVariableOp,^simple_rnn_cell_163/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_163/BiasAdd/ReadVariableOp*simple_rnn_cell_163/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_163/MatMul/ReadVariableOp)simple_rnn_cell_163/MatMul/ReadVariableOp2Z
+simple_rnn_cell_163/MatMul_1/ReadVariableOp+simple_rnn_cell_163/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?-
?
while_body_10851884
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_162_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_162_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_162_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_162_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_162/MatMul/ReadVariableOp?1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_162_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_162/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_162/BiasAddBiasAdd*while/simple_rnn_cell_162/MatMul:product:08while/simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_162/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_162/addAddV2*while/simple_rnn_cell_162/BiasAdd:output:0,while/simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_162/TanhTanh!while/simple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_162/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_162/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_162/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_162/MatMul/ReadVariableOp2^while/simple_rnn_cell_162/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_162_biasadd_readvariableop_resource;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_162_matmul_1_readvariableop_resource<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_162_matmul_readvariableop_resource:while_simple_rnn_cell_162_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_162/MatMul/ReadVariableOp/while/simple_rnn_cell_162/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?	
f
G__inference_dropout_3_layer_call_and_return_conditional_losses_10852153

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????dC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????do
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????di
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????dY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?>
?
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10853865
inputs_0D
2simple_rnn_cell_163_matmul_readvariableop_resource:PdA
3simple_rnn_cell_163_biasadd_readvariableop_resource:dF
4simple_rnn_cell_163_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_163/BiasAdd/ReadVariableOp?)simple_rnn_cell_163/MatMul/ReadVariableOp?+simple_rnn_cell_163/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
)simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_163_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_163/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_163_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_163/BiasAddBiasAdd$simple_rnn_cell_163/MatMul:product:02simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_163_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_163/MatMul_1MatMulzeros:output:03simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_163/addAddV2$simple_rnn_cell_163/BiasAdd:output:0&simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_163/TanhTanhsimple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_163_matmul_readvariableop_resource3simple_rnn_cell_163_biasadd_readvariableop_resource4simple_rnn_cell_163_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10853799*
condR
while_cond_10853798*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_163/BiasAdd/ReadVariableOp*^simple_rnn_cell_163/MatMul/ReadVariableOp,^simple_rnn_cell_163/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2X
*simple_rnn_cell_163/BiasAdd/ReadVariableOp*simple_rnn_cell_163/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_163/MatMul/ReadVariableOp)simple_rnn_cell_163/MatMul/ReadVariableOp2Z
+simple_rnn_cell_163/MatMul_1/ReadVariableOp+simple_rnn_cell_163/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?

?
 simple_rnn_2_while_cond_108526666
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_28
4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1P
Lsimple_rnn_2_while_simple_rnn_2_while_cond_10852666___redundant_placeholder0P
Lsimple_rnn_2_while_simple_rnn_2_while_cond_10852666___redundant_placeholder1P
Lsimple_rnn_2_while_simple_rnn_2_while_cond_10852666___redundant_placeholder2P
Lsimple_rnn_2_while_simple_rnn_2_while_cond_10852666___redundant_placeholder3
simple_rnn_2_while_identity
?
simple_rnn_2/while/LessLesssimple_rnn_2_while_placeholder4simple_rnn_2_while_less_simple_rnn_2_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852104

inputs'
simple_rnn_2_10851951:P#
simple_rnn_2_10851953:P'
simple_rnn_2_10851955:PP'
simple_rnn_3_10852073:Pd#
simple_rnn_3_10852075:d'
simple_rnn_3_10852077:dd"
dense_1_10852098:d
dense_1_10852100:
identity??dense_1/StatefulPartitionedCall?$simple_rnn_2/StatefulPartitionedCall?$simple_rnn_3/StatefulPartitionedCall?
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_2_10851951simple_rnn_2_10851953simple_rnn_2_10851955*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10851950?
dropout_2/PartitionedCallPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_10851963?
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0simple_rnn_3_10852073simple_rnn_3_10852075simple_rnn_3_10852077*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10852072?
dropout_3/PartitionedCallPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_10852085?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_1_10852098dense_1_10852100*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_10852097w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_10851471
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10851471___redundant_placeholder06
2while_while_cond_10851471___redundant_placeholder16
2while_while_cond_10851471___redundant_placeholder26
2while_while_cond_10851471___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?
?
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10851592

inputs

states0
matmul_readvariableop_resource:Pd-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????dG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????dY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????P:?????????d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates
?H
?
!__inference__traced_save_10854367
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_simple_rnn_2_simple_rnn_cell_2_kernel_read_readvariableopN
Jsavev2_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_2_simple_rnn_cell_2_bias_read_readvariableopD
@savev2_simple_rnn_3_simple_rnn_cell_3_kernel_read_readvariableopN
Jsavev2_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_3_simple_rnn_cell_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_2_simple_rnn_cell_2_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_2_simple_rnn_cell_2_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_3_simple_rnn_cell_3_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_3_simple_rnn_cell_3_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_2_simple_rnn_cell_2_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_2_simple_rnn_cell_2_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_3_simple_rnn_cell_3_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_3_simple_rnn_cell_3_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_simple_rnn_2_simple_rnn_cell_2_kernel_read_readvariableopJsavev2_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_read_readvariableop>savev2_simple_rnn_2_simple_rnn_cell_2_bias_read_readvariableop@savev2_simple_rnn_3_simple_rnn_cell_3_kernel_read_readvariableopJsavev2_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_read_readvariableop>savev2_simple_rnn_3_simple_rnn_cell_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableopGsavev2_adam_simple_rnn_2_simple_rnn_cell_2_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_2_simple_rnn_cell_2_bias_m_read_readvariableopGsavev2_adam_simple_rnn_3_simple_rnn_cell_3_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_3_simple_rnn_cell_3_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopGsavev2_adam_simple_rnn_2_simple_rnn_cell_2_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_2_simple_rnn_cell_2_bias_v_read_readvariableopGsavev2_adam_simple_rnn_3_simple_rnn_cell_3_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_3_simple_rnn_cell_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :d:: : : : : :P:PP:P:Pd:dd:d: : :d::P:PP:P:Pd:dd:d:d::P:PP:P:Pd:dd:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:P:$	 

_output_shapes

:PP: 


_output_shapes
:P:$ 

_output_shapes

:Pd:$ 

_output_shapes

:dd: 

_output_shapes
:d:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:Pd:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:Pd:$ 

_output_shapes

:dd: 

_output_shapes
:d: 

_output_shapes
: 
?
?
/__inference_simple_rnn_3_layer_call_fn_10853616
inputs_0
unknown:Pd
	unknown_0:d
	unknown_1:dd
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10851668o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?

?
6__inference_simple_rnn_cell_163_layer_call_fn_10854217

inputs
states_0
unknown:Pd
	unknown_0:d
	unknown_1:dd
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10851712o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????P:?????????d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0
??
?	
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852845

inputsQ
?simple_rnn_2_simple_rnn_cell_162_matmul_readvariableop_resource:PN
@simple_rnn_2_simple_rnn_cell_162_biasadd_readvariableop_resource:PS
Asimple_rnn_2_simple_rnn_cell_162_matmul_1_readvariableop_resource:PPQ
?simple_rnn_3_simple_rnn_cell_163_matmul_readvariableop_resource:PdN
@simple_rnn_3_simple_rnn_cell_163_biasadd_readvariableop_resource:dS
Asimple_rnn_3_simple_rnn_cell_163_matmul_1_readvariableop_resource:dd8
&dense_1_matmul_readvariableop_resource:d5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?7simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOp?6simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOp?8simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp?simple_rnn_2/while?7simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOp?6simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOp?8simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp?simple_rnn_3/whileH
simple_rnn_2/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_2/strided_sliceStridedSlicesimple_rnn_2/Shape:output:0)simple_rnn_2/strided_slice/stack:output:0+simple_rnn_2/strided_slice/stack_1:output:0+simple_rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
simple_rnn_2/zeros/packedPack#simple_rnn_2/strided_slice:output:0$simple_rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_2/zerosFill"simple_rnn_2/zeros/packed:output:0!simple_rnn_2/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pp
simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_2/transpose	Transposeinputs$simple_rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????^
simple_rnn_2/Shape_1Shapesimple_rnn_2/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_2/strided_slice_1StridedSlicesimple_rnn_2/Shape_1:output:0+simple_rnn_2/strided_slice_1/stack:output:0-simple_rnn_2/strided_slice_1/stack_1:output:0-simple_rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_2/TensorArrayV2TensorListReserve1simple_rnn_2/TensorArrayV2/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_2/transpose:y:0Ksimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_2/strided_slice_2StridedSlicesimple_rnn_2/transpose:y:0+simple_rnn_2/strided_slice_2/stack:output:0-simple_rnn_2/strided_slice_2/stack_1:output:0-simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
6simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp?simple_rnn_2_simple_rnn_cell_162_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
'simple_rnn_2/simple_rnn_cell_162/MatMulMatMul%simple_rnn_2/strided_slice_2:output:0>simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
7simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_2_simple_rnn_cell_162_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
(simple_rnn_2/simple_rnn_cell_162/BiasAddBiasAdd1simple_rnn_2/simple_rnn_cell_162/MatMul:product:0?simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_2_simple_rnn_cell_162_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
)simple_rnn_2/simple_rnn_cell_162/MatMul_1MatMulsimple_rnn_2/zeros:output:0@simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
$simple_rnn_2/simple_rnn_cell_162/addAddV21simple_rnn_2/simple_rnn_cell_162/BiasAdd:output:03simple_rnn_2/simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
%simple_rnn_2/simple_rnn_cell_162/TanhTanh(simple_rnn_2/simple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????P{
*simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
simple_rnn_2/TensorArrayV2_1TensorListReserve3simple_rnn_2/TensorArrayV2_1/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_2/whileWhile(simple_rnn_2/while/loop_counter:output:0.simple_rnn_2/while/maximum_iterations:output:0simple_rnn_2/time:output:0%simple_rnn_2/TensorArrayV2_1:handle:0simple_rnn_2/zeros:output:0%simple_rnn_2/strided_slice_1:output:0Dsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_2_simple_rnn_cell_162_matmul_readvariableop_resource@simple_rnn_2_simple_rnn_cell_162_biasadd_readvariableop_resourceAsimple_rnn_2_simple_rnn_cell_162_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_2_while_body_10852667*,
cond$R"
 simple_rnn_2_while_cond_10852666*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_2/while:output:3Fsimple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0u
"simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_2/strided_slice_3StridedSlice8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_2/strided_slice_3/stack:output:0-simple_rnn_2/strided_slice_3/stack_1:output:0-simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maskr
simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_2/transpose_1	Transpose8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pr
dropout_2/IdentityIdentitysimple_rnn_2/transpose_1:y:0*
T0*+
_output_shapes
:?????????P]
simple_rnn_3/ShapeShapedropout_2/Identity:output:0*
T0*
_output_shapes
:j
 simple_rnn_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_3/strided_sliceStridedSlicesimple_rnn_3/Shape:output:0)simple_rnn_3/strided_slice/stack:output:0+simple_rnn_3/strided_slice/stack_1:output:0+simple_rnn_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
simple_rnn_3/zeros/packedPack#simple_rnn_3/strided_slice:output:0$simple_rnn_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_3/zerosFill"simple_rnn_3/zeros/packed:output:0!simple_rnn_3/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dp
simple_rnn_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_3/transpose	Transposedropout_2/Identity:output:0$simple_rnn_3/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P^
simple_rnn_3/Shape_1Shapesimple_rnn_3/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_3/strided_slice_1StridedSlicesimple_rnn_3/Shape_1:output:0+simple_rnn_3/strided_slice_1/stack:output:0-simple_rnn_3/strided_slice_1/stack_1:output:0-simple_rnn_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_3/TensorArrayV2TensorListReserve1simple_rnn_3/TensorArrayV2/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
4simple_rnn_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_3/transpose:y:0Ksimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_3/strided_slice_2StridedSlicesimple_rnn_3/transpose:y:0+simple_rnn_3/strided_slice_2/stack:output:0-simple_rnn_3/strided_slice_2/stack_1:output:0-simple_rnn_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
6simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp?simple_rnn_3_simple_rnn_cell_163_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
'simple_rnn_3/simple_rnn_cell_163/MatMulMatMul%simple_rnn_3/strided_slice_2:output:0>simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
7simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_3_simple_rnn_cell_163_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
(simple_rnn_3/simple_rnn_cell_163/BiasAddBiasAdd1simple_rnn_3/simple_rnn_cell_163/MatMul:product:0?simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_3_simple_rnn_cell_163_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
)simple_rnn_3/simple_rnn_cell_163/MatMul_1MatMulsimple_rnn_3/zeros:output:0@simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
$simple_rnn_3/simple_rnn_cell_163/addAddV21simple_rnn_3/simple_rnn_cell_163/BiasAdd:output:03simple_rnn_3/simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
%simple_rnn_3/simple_rnn_cell_163/TanhTanh(simple_rnn_3/simple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????d{
*simple_rnn_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
simple_rnn_3/TensorArrayV2_1TensorListReserve3simple_rnn_3/TensorArrayV2_1/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_3/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_3/whileWhile(simple_rnn_3/while/loop_counter:output:0.simple_rnn_3/while/maximum_iterations:output:0simple_rnn_3/time:output:0%simple_rnn_3/TensorArrayV2_1:handle:0simple_rnn_3/zeros:output:0%simple_rnn_3/strided_slice_1:output:0Dsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_3_simple_rnn_cell_163_matmul_readvariableop_resource@simple_rnn_3_simple_rnn_cell_163_biasadd_readvariableop_resourceAsimple_rnn_3_simple_rnn_cell_163_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_3_while_body_10852772*,
cond$R"
 simple_rnn_3_while_cond_10852771*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
=simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
/simple_rnn_3/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_3/while:output:3Fsimple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0u
"simple_rnn_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_3/strided_slice_3StridedSlice8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_3/strided_slice_3/stack:output:0-simple_rnn_3/strided_slice_3/stack_1:output:0-simple_rnn_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maskr
simple_rnn_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_3/transpose_1	Transpose8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dw
dropout_3/IdentityIdentity%simple_rnn_3/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_1/MatMulMatMuldropout_3/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp8^simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOp7^simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOp9^simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp^simple_rnn_2/while8^simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOp7^simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOp9^simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp^simple_rnn_3/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2r
7simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOp7simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOp2p
6simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOp6simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOp2t
8simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp8simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp2(
simple_rnn_2/whilesimple_rnn_2/while2r
7simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOp7simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOp2p
6simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOp6simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOp2t
8simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp8simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp2(
simple_rnn_3/whilesimple_rnn_3/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
while_body_10853404
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_162_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_162_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_162_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_162_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_162/MatMul/ReadVariableOp?1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_162_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_162/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_162/BiasAddBiasAdd*while/simple_rnn_cell_162/MatMul:product:08while/simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_162/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_162/addAddV2*while/simple_rnn_cell_162/BiasAdd:output:0,while/simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_162/TanhTanh!while/simple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_162/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_162/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_162/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_162/MatMul/ReadVariableOp2^while/simple_rnn_cell_162/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_162_biasadd_readvariableop_resource;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_162_matmul_1_readvariableop_resource<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_162_matmul_readvariableop_resource:while_simple_rnn_cell_162_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_162/MatMul/ReadVariableOp/while/simple_rnn_cell_162/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_10853187
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10853187___redundant_placeholder06
2while_while_cond_10853187___redundant_placeholder16
2while_while_cond_10853187___redundant_placeholder26
2while_while_cond_10853187___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?!
?
while_body_10851313
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_162_10851335_0:P2
$while_simple_rnn_cell_162_10851337_0:P6
$while_simple_rnn_cell_162_10851339_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_162_10851335:P0
"while_simple_rnn_cell_162_10851337:P4
"while_simple_rnn_cell_162_10851339:PP??1while/simple_rnn_cell_162/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/simple_rnn_cell_162/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_162_10851335_0$while_simple_rnn_cell_162_10851337_0$while_simple_rnn_cell_162_10851339_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????P:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10851300?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_162/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :????
while/Identity_4Identity:while/simple_rnn_cell_162/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp2^while/simple_rnn_cell_162/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_162_10851335$while_simple_rnn_cell_162_10851335_0"J
"while_simple_rnn_cell_162_10851337$while_simple_rnn_cell_162_10851337_0"J
"while_simple_rnn_cell_162_10851339$while_simple_rnn_cell_162_10851339_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2f
1while/simple_rnn_cell_162/StatefulPartitionedCall1while/simple_rnn_cell_162/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_10852363
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10852363___redundant_placeholder06
2while_while_cond_10852363___redundant_placeholder16
2while_while_cond_10852363___redundant_placeholder26
2while_while_cond_10852363___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_10853295
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10853295___redundant_placeholder06
2while_while_cond_10853295___redundant_placeholder16
2while_while_cond_10853295___redundant_placeholder26
2while_while_cond_10853295___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?
?
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852577
simple_rnn_2_input'
simple_rnn_2_10852555:P#
simple_rnn_2_10852557:P'
simple_rnn_2_10852559:PP'
simple_rnn_3_10852563:Pd#
simple_rnn_3_10852565:d'
simple_rnn_3_10852567:dd"
dense_1_10852571:d
dense_1_10852573:
identity??dense_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?$simple_rnn_2/StatefulPartitionedCall?$simple_rnn_3/StatefulPartitionedCall?
$simple_rnn_2/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputsimple_rnn_2_10852555simple_rnn_2_10852557simple_rnn_2_10852559*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10852430?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_10852306?
$simple_rnn_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0simple_rnn_3_10852563simple_rnn_3_10852565simple_rnn_3_10852567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10852277?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_3/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_10852153?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_1_10852571dense_1_10852573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_10852097w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall%^simple_rnn_2/StatefulPartitionedCall%^simple_rnn_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2L
$simple_rnn_2/StatefulPartitionedCall$simple_rnn_2/StatefulPartitionedCall2L
$simple_rnn_3/StatefulPartitionedCall$simple_rnn_3/StatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_2_input
?-
?
while_body_10852006
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_163_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_163_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_163_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_163_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_163/MatMul/ReadVariableOp?1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_163_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_163/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_163/BiasAddBiasAdd*while/simple_rnn_cell_163/MatMul:product:08while/simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_163/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_163/addAddV2*while/simple_rnn_cell_163/BiasAdd:output:0,while/simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_163/TanhTanh!while/simple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_163/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_163/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_163/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_163/MatMul/ReadVariableOp2^while/simple_rnn_cell_163/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_163_biasadd_readvariableop_resource;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_163_matmul_1_readvariableop_resource<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_163_matmul_readvariableop_resource:while_simple_rnn_cell_163_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_163/MatMul/ReadVariableOp/while/simple_rnn_cell_163/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?9
?
 simple_rnn_2_while_body_108526676
2simple_rnn_2_while_simple_rnn_2_while_loop_counter<
8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations"
simple_rnn_2_while_placeholder$
 simple_rnn_2_while_placeholder_1$
 simple_rnn_2_while_placeholder_25
1simple_rnn_2_while_simple_rnn_2_strided_slice_1_0q
msimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resource_0:PV
Hsimple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resource_0:P[
Isimple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0:PP
simple_rnn_2_while_identity!
simple_rnn_2_while_identity_1!
simple_rnn_2_while_identity_2!
simple_rnn_2_while_identity_3!
simple_rnn_2_while_identity_43
/simple_rnn_2_while_simple_rnn_2_strided_slice_1o
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resource:PT
Fsimple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resource:PY
Gsimple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resource:PP??=simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOp?<simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOp?>simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp?
Dsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
6simple_rnn_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_2_while_placeholderMsimple_rnn_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
<simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
-simple_rnn_2/while/simple_rnn_cell_162/MatMulMatMul=simple_rnn_2/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
=simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
.simple_rnn_2/while/simple_rnn_cell_162/BiasAddBiasAdd7simple_rnn_2/while/simple_rnn_cell_162/MatMul:product:0Esimple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
>simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
/simple_rnn_2/while/simple_rnn_cell_162/MatMul_1MatMul simple_rnn_2_while_placeholder_2Fsimple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_2/while/simple_rnn_cell_162/addAddV27simple_rnn_2/while/simple_rnn_cell_162/BiasAdd:output:09simple_rnn_2/while/simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_2/while/simple_rnn_cell_162/TanhTanh.simple_rnn_2/while/simple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????P?
7simple_rnn_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_2_while_placeholder_1simple_rnn_2_while_placeholder/simple_rnn_2/while/simple_rnn_cell_162/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_2/while/addAddV2simple_rnn_2_while_placeholder!simple_rnn_2/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_2/while/add_1AddV22simple_rnn_2_while_simple_rnn_2_while_loop_counter#simple_rnn_2/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_2/while/IdentityIdentitysimple_rnn_2/while/add_1:z:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_2/while/Identity_1Identity8simple_rnn_2_while_simple_rnn_2_while_maximum_iterations^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_2/while/Identity_2Identitysimple_rnn_2/while/add:z:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_2/while/Identity_3IdentityGsimple_rnn_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_2/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_2/while/Identity_4Identity/simple_rnn_2/while/simple_rnn_cell_162/Tanh:y:0^simple_rnn_2/while/NoOp*
T0*'
_output_shapes
:?????????P?
simple_rnn_2/while/NoOpNoOp>^simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOp=^simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOp?^simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_2_while_identity$simple_rnn_2/while/Identity:output:0"G
simple_rnn_2_while_identity_1&simple_rnn_2/while/Identity_1:output:0"G
simple_rnn_2_while_identity_2&simple_rnn_2/while/Identity_2:output:0"G
simple_rnn_2_while_identity_3&simple_rnn_2/while/Identity_3:output:0"G
simple_rnn_2_while_identity_4&simple_rnn_2/while/Identity_4:output:0"d
/simple_rnn_2_while_simple_rnn_2_strided_slice_11simple_rnn_2_while_simple_rnn_2_strided_slice_1_0"?
Fsimple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resourceHsimple_rnn_2_while_simple_rnn_cell_162_biasadd_readvariableop_resource_0"?
Gsimple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resourceIsimple_rnn_2_while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0"?
Esimple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resourceGsimple_rnn_2_while_simple_rnn_cell_162_matmul_readvariableop_resource_0"?
ksimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensormsimple_rnn_2_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2~
=simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOp=simple_rnn_2/while/simple_rnn_cell_162/BiasAdd/ReadVariableOp2|
<simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOp<simple_rnn_2/while/simple_rnn_cell_162/MatMul/ReadVariableOp2?
>simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp>simple_rnn_2/while/simple_rnn_cell_162/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_10851312
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10851312___redundant_placeholder06
2while_while_cond_10851312___redundant_placeholder16
2while_while_cond_10851312___redundant_placeholder26
2while_while_cond_10851312___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_10853798
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10853798___redundant_placeholder06
2while_while_cond_10853798___redundant_placeholder16
2while_while_cond_10853798___redundant_placeholder26
2while_while_cond_10853798___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_10852210
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10852210___redundant_placeholder06
2while_while_cond_10852210___redundant_placeholder16
2while_while_cond_10852210___redundant_placeholder26
2while_while_cond_10852210___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_10851883
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10851883___redundant_placeholder06
2while_while_cond_10851883___redundant_placeholder16
2while_while_cond_10851883___redundant_placeholder26
2while_while_cond_10851883___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?
?
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10854189

inputs
states_00
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:P2
 matmul_1_readvariableop_resource:PP
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Px
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????PG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????PW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????PY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????P?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????P: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????P
"
_user_specified_name
states/0
?>
?
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10853757
inputs_0D
2simple_rnn_cell_163_matmul_readvariableop_resource:PdA
3simple_rnn_cell_163_biasadd_readvariableop_resource:dF
4simple_rnn_cell_163_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_163/BiasAdd/ReadVariableOp?)simple_rnn_cell_163/MatMul/ReadVariableOp?+simple_rnn_cell_163/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
)simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_163_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_163/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_163_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_163/BiasAddBiasAdd$simple_rnn_cell_163/MatMul:product:02simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_163_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_163/MatMul_1MatMulzeros:output:03simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_163/addAddV2$simple_rnn_cell_163/BiasAdd:output:0&simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_163/TanhTanhsimple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_163_matmul_readvariableop_resource3simple_rnn_cell_163_biasadd_readvariableop_resource4simple_rnn_cell_163_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10853691*
condR
while_cond_10853690*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_163/BiasAdd/ReadVariableOp*^simple_rnn_cell_163/MatMul/ReadVariableOp,^simple_rnn_cell_163/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2X
*simple_rnn_cell_163/BiasAdd/ReadVariableOp*simple_rnn_cell_163/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_163/MatMul/ReadVariableOp)simple_rnn_cell_163/MatMul/ReadVariableOp2Z
+simple_rnn_cell_163/MatMul_1/ReadVariableOp+simple_rnn_cell_163/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?
?
/__inference_simple_rnn_3_layer_call_fn_10853638

inputs
unknown:Pd
	unknown_0:d
	unknown_1:dd
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10852072o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
/__inference_simple_rnn_2_layer_call_fn_10853124
inputs_0
unknown:P
	unknown_0:P
	unknown_1:PP
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10851535|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?-
?
while_body_10853188
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_162_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_162_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_162_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_162_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_162/MatMul/ReadVariableOp?1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_162_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_162/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_162/BiasAddBiasAdd*while/simple_rnn_cell_162/MatMul:product:08while/simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_162/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_162/addAddV2*while/simple_rnn_cell_162/BiasAdd:output:0,while/simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_162/TanhTanh!while/simple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_162/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_162/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_162/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_162/MatMul/ReadVariableOp2^while/simple_rnn_cell_162/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_162_biasadd_readvariableop_resource;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_162_matmul_1_readvariableop_resource<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_162_matmul_readvariableop_resource:while_simple_rnn_cell_162_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_162/MatMul/ReadVariableOp/while/simple_rnn_cell_162/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?9
?
 simple_rnn_3_while_body_108529996
2simple_rnn_3_while_simple_rnn_3_while_loop_counter<
8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations"
simple_rnn_3_while_placeholder$
 simple_rnn_3_while_placeholder_1$
 simple_rnn_3_while_placeholder_25
1simple_rnn_3_while_simple_rnn_3_strided_slice_1_0q
msimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resource_0:PdV
Hsimple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resource_0:d[
Isimple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0:dd
simple_rnn_3_while_identity!
simple_rnn_3_while_identity_1!
simple_rnn_3_while_identity_2!
simple_rnn_3_while_identity_3!
simple_rnn_3_while_identity_43
/simple_rnn_3_while_simple_rnn_3_strided_slice_1o
ksimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resource:PdT
Fsimple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resource:dY
Gsimple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resource:dd??=simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOp?<simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOp?>simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp?
Dsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
6simple_rnn_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_3_while_placeholderMsimple_rnn_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
<simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
-simple_rnn_3/while/simple_rnn_cell_163/MatMulMatMul=simple_rnn_3/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
=simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
.simple_rnn_3/while/simple_rnn_cell_163/BiasAddBiasAdd7simple_rnn_3/while/simple_rnn_cell_163/MatMul:product:0Esimple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
>simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
/simple_rnn_3/while/simple_rnn_cell_163/MatMul_1MatMul simple_rnn_3_while_placeholder_2Fsimple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_3/while/simple_rnn_cell_163/addAddV27simple_rnn_3/while/simple_rnn_cell_163/BiasAdd:output:09simple_rnn_3/while/simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_3/while/simple_rnn_cell_163/TanhTanh.simple_rnn_3/while/simple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????d?
7simple_rnn_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_3_while_placeholder_1simple_rnn_3_while_placeholder/simple_rnn_3/while/simple_rnn_cell_163/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_3/while/addAddV2simple_rnn_3_while_placeholder!simple_rnn_3/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_3/while/add_1AddV22simple_rnn_3_while_simple_rnn_3_while_loop_counter#simple_rnn_3/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_3/while/IdentityIdentitysimple_rnn_3/while/add_1:z:0^simple_rnn_3/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_3/while/Identity_1Identity8simple_rnn_3_while_simple_rnn_3_while_maximum_iterations^simple_rnn_3/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_3/while/Identity_2Identitysimple_rnn_3/while/add:z:0^simple_rnn_3/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_3/while/Identity_3IdentityGsimple_rnn_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_3/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_3/while/Identity_4Identity/simple_rnn_3/while/simple_rnn_cell_163/Tanh:y:0^simple_rnn_3/while/NoOp*
T0*'
_output_shapes
:?????????d?
simple_rnn_3/while/NoOpNoOp>^simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOp=^simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOp?^simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_3_while_identity$simple_rnn_3/while/Identity:output:0"G
simple_rnn_3_while_identity_1&simple_rnn_3/while/Identity_1:output:0"G
simple_rnn_3_while_identity_2&simple_rnn_3/while/Identity_2:output:0"G
simple_rnn_3_while_identity_3&simple_rnn_3/while/Identity_3:output:0"G
simple_rnn_3_while_identity_4&simple_rnn_3/while/Identity_4:output:0"d
/simple_rnn_3_while_simple_rnn_3_strided_slice_11simple_rnn_3_while_simple_rnn_3_strided_slice_1_0"?
Fsimple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resourceHsimple_rnn_3_while_simple_rnn_cell_163_biasadd_readvariableop_resource_0"?
Gsimple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resourceIsimple_rnn_3_while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0"?
Esimple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resourceGsimple_rnn_3_while_simple_rnn_cell_163_matmul_readvariableop_resource_0"?
ksimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensormsimple_rnn_3_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2~
=simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOp=simple_rnn_3/while/simple_rnn_cell_163/BiasAdd/ReadVariableOp2|
<simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOp<simple_rnn_3/while/simple_rnn_cell_163/MatMul/ReadVariableOp2?
>simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp>simple_rnn_3/while/simple_rnn_cell_163/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
??
?	
J__inference_sequential_1_layer_call_and_return_conditional_losses_10853079

inputsQ
?simple_rnn_2_simple_rnn_cell_162_matmul_readvariableop_resource:PN
@simple_rnn_2_simple_rnn_cell_162_biasadd_readvariableop_resource:PS
Asimple_rnn_2_simple_rnn_cell_162_matmul_1_readvariableop_resource:PPQ
?simple_rnn_3_simple_rnn_cell_163_matmul_readvariableop_resource:PdN
@simple_rnn_3_simple_rnn_cell_163_biasadd_readvariableop_resource:dS
Asimple_rnn_3_simple_rnn_cell_163_matmul_1_readvariableop_resource:dd8
&dense_1_matmul_readvariableop_resource:d5
'dense_1_biasadd_readvariableop_resource:
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?7simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOp?6simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOp?8simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp?simple_rnn_2/while?7simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOp?6simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOp?8simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp?simple_rnn_3/whileH
simple_rnn_2/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_2/strided_sliceStridedSlicesimple_rnn_2/Shape:output:0)simple_rnn_2/strided_slice/stack:output:0+simple_rnn_2/strided_slice/stack_1:output:0+simple_rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
simple_rnn_2/zeros/packedPack#simple_rnn_2/strided_slice:output:0$simple_rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_2/zerosFill"simple_rnn_2/zeros/packed:output:0!simple_rnn_2/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pp
simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_2/transpose	Transposeinputs$simple_rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????^
simple_rnn_2/Shape_1Shapesimple_rnn_2/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_2/strided_slice_1StridedSlicesimple_rnn_2/Shape_1:output:0+simple_rnn_2/strided_slice_1/stack:output:0-simple_rnn_2/strided_slice_1/stack_1:output:0-simple_rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_2/TensorArrayV2TensorListReserve1simple_rnn_2/TensorArrayV2/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
4simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_2/transpose:y:0Ksimple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_2/strided_slice_2StridedSlicesimple_rnn_2/transpose:y:0+simple_rnn_2/strided_slice_2/stack:output:0-simple_rnn_2/strided_slice_2/stack_1:output:0-simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
6simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp?simple_rnn_2_simple_rnn_cell_162_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
'simple_rnn_2/simple_rnn_cell_162/MatMulMatMul%simple_rnn_2/strided_slice_2:output:0>simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
7simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_2_simple_rnn_cell_162_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
(simple_rnn_2/simple_rnn_cell_162/BiasAddBiasAdd1simple_rnn_2/simple_rnn_cell_162/MatMul:product:0?simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_2_simple_rnn_cell_162_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
)simple_rnn_2/simple_rnn_cell_162/MatMul_1MatMulsimple_rnn_2/zeros:output:0@simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
$simple_rnn_2/simple_rnn_cell_162/addAddV21simple_rnn_2/simple_rnn_cell_162/BiasAdd:output:03simple_rnn_2/simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
%simple_rnn_2/simple_rnn_cell_162/TanhTanh(simple_rnn_2/simple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????P{
*simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
simple_rnn_2/TensorArrayV2_1TensorListReserve3simple_rnn_2/TensorArrayV2_1/element_shape:output:0%simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_2/whileWhile(simple_rnn_2/while/loop_counter:output:0.simple_rnn_2/while/maximum_iterations:output:0simple_rnn_2/time:output:0%simple_rnn_2/TensorArrayV2_1:handle:0simple_rnn_2/zeros:output:0%simple_rnn_2/strided_slice_1:output:0Dsimple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_2_simple_rnn_cell_162_matmul_readvariableop_resource@simple_rnn_2_simple_rnn_cell_162_biasadd_readvariableop_resourceAsimple_rnn_2_simple_rnn_cell_162_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_2_while_body_10852887*,
cond$R"
 simple_rnn_2_while_cond_10852886*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
=simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_2/while:output:3Fsimple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0u
"simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_2/strided_slice_3StridedSlice8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_2/strided_slice_3/stack:output:0-simple_rnn_2/strided_slice_3/stack_1:output:0-simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maskr
simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_2/transpose_1	Transpose8simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_2/dropout/MulMulsimple_rnn_2/transpose_1:y:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:?????????Pc
dropout_2/dropout/ShapeShapesimple_rnn_2/transpose_1:y:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????P?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????P?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
simple_rnn_3/ShapeShapedropout_2/dropout/Mul_1:z:0*
T0*
_output_shapes
:j
 simple_rnn_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_3/strided_sliceStridedSlicesimple_rnn_3/Shape:output:0)simple_rnn_3/strided_slice/stack:output:0+simple_rnn_3/strided_slice/stack_1:output:0+simple_rnn_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
simple_rnn_3/zeros/packedPack#simple_rnn_3/strided_slice:output:0$simple_rnn_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_3/zerosFill"simple_rnn_3/zeros/packed:output:0!simple_rnn_3/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dp
simple_rnn_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_3/transpose	Transposedropout_2/dropout/Mul_1:z:0$simple_rnn_3/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P^
simple_rnn_3/Shape_1Shapesimple_rnn_3/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_3/strided_slice_1StridedSlicesimple_rnn_3/Shape_1:output:0+simple_rnn_3/strided_slice_1/stack:output:0-simple_rnn_3/strided_slice_1/stack_1:output:0-simple_rnn_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_3/TensorArrayV2TensorListReserve1simple_rnn_3/TensorArrayV2/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
4simple_rnn_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_3/transpose:y:0Ksimple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_3/strided_slice_2StridedSlicesimple_rnn_3/transpose:y:0+simple_rnn_3/strided_slice_2/stack:output:0-simple_rnn_3/strided_slice_2/stack_1:output:0-simple_rnn_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
6simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp?simple_rnn_3_simple_rnn_cell_163_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
'simple_rnn_3/simple_rnn_cell_163/MatMulMatMul%simple_rnn_3/strided_slice_2:output:0>simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
7simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_3_simple_rnn_cell_163_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
(simple_rnn_3/simple_rnn_cell_163/BiasAddBiasAdd1simple_rnn_3/simple_rnn_cell_163/MatMul:product:0?simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_3_simple_rnn_cell_163_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
)simple_rnn_3/simple_rnn_cell_163/MatMul_1MatMulsimple_rnn_3/zeros:output:0@simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
$simple_rnn_3/simple_rnn_cell_163/addAddV21simple_rnn_3/simple_rnn_cell_163/BiasAdd:output:03simple_rnn_3/simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
%simple_rnn_3/simple_rnn_cell_163/TanhTanh(simple_rnn_3/simple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????d{
*simple_rnn_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
simple_rnn_3/TensorArrayV2_1TensorListReserve3simple_rnn_3/TensorArrayV2_1/element_shape:output:0%simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_3/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_3/whileWhile(simple_rnn_3/while/loop_counter:output:0.simple_rnn_3/while/maximum_iterations:output:0simple_rnn_3/time:output:0%simple_rnn_3/TensorArrayV2_1:handle:0simple_rnn_3/zeros:output:0%simple_rnn_3/strided_slice_1:output:0Dsimple_rnn_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_3_simple_rnn_cell_163_matmul_readvariableop_resource@simple_rnn_3_simple_rnn_cell_163_biasadd_readvariableop_resourceAsimple_rnn_3_simple_rnn_cell_163_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_3_while_body_10852999*,
cond$R"
 simple_rnn_3_while_cond_10852998*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
=simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
/simple_rnn_3/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_3/while:output:3Fsimple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0u
"simple_rnn_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_3/strided_slice_3StridedSlice8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_3/strided_slice_3/stack:output:0-simple_rnn_3/strided_slice_3/stack_1:output:0-simple_rnn_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maskr
simple_rnn_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_3/transpose_1	Transpose8simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_3/dropout/MulMul%simple_rnn_3/strided_slice_3:output:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????dl
dropout_3/dropout/ShapeShape%simple_rnn_3/strided_slice_3:output:0*
T0*
_output_shapes
:?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_1/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp8^simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOp7^simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOp9^simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp^simple_rnn_2/while8^simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOp7^simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOp9^simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp^simple_rnn_3/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2r
7simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOp7simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOp2p
6simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOp6simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOp2t
8simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp8simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp2(
simple_rnn_2/whilesimple_rnn_2/while2r
7simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOp7simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOp2p
6simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOp6simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOp2t
8simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp8simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp2(
simple_rnn_3/whilesimple_rnn_3/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
while_body_10853799
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_163_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_163_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_163_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_163_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_163/MatMul/ReadVariableOp?1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_163_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_163/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_163/BiasAddBiasAdd*while/simple_rnn_cell_163/MatMul:product:08while/simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_163/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_163/addAddV2*while/simple_rnn_cell_163/BiasAdd:output:0,while/simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_163/TanhTanh!while/simple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_163/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_163/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_163/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_163/MatMul/ReadVariableOp2^while/simple_rnn_cell_163/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_163_biasadd_readvariableop_resource;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_163_matmul_1_readvariableop_resource<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_163_matmul_readvariableop_resource:while_simple_rnn_cell_163_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_163/MatMul/ReadVariableOp/while/simple_rnn_cell_163/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?	
?
/__inference_sequential_1_layer_call_fn_10852625

inputs
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852487o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
6__inference_simple_rnn_cell_163_layer_call_fn_10854203

inputs
states_0
unknown:Pd
	unknown_0:d
	unknown_1:dd
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10851592o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????P:?????????d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0
?
?
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10851420

inputs

states0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:P2
 matmul_1_readvariableop_resource:PP
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Px
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????PG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????PW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????PY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????P?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????P: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????P
 
_user_specified_namestates
?

?
6__inference_simple_rnn_cell_162_layer_call_fn_10854141

inputs
states_0
unknown:P
	unknown_0:P
	unknown_1:PP
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????P:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10851300o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Pq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????P
"
_user_specified_name
states/0
?-
?
while_body_10853691
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_163_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_163_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_163_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_163_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_163/MatMul/ReadVariableOp?1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_163_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_163/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_163/BiasAddBiasAdd*while/simple_rnn_cell_163/MatMul:product:08while/simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_163/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_163/addAddV2*while/simple_rnn_cell_163/BiasAdd:output:0,while/simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_163/TanhTanh!while/simple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_163/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_163/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_163/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_163/MatMul/ReadVariableOp2^while/simple_rnn_cell_163/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_163_biasadd_readvariableop_resource;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_163_matmul_1_readvariableop_resource<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_163_matmul_readvariableop_resource:while_simple_rnn_cell_163_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_163/MatMul/ReadVariableOp/while/simple_rnn_cell_163/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?=
?
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10851950

inputsD
2simple_rnn_cell_162_matmul_readvariableop_resource:PA
3simple_rnn_cell_162_biasadd_readvariableop_resource:PF
4simple_rnn_cell_162_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_162/BiasAdd/ReadVariableOp?)simple_rnn_cell_162/MatMul/ReadVariableOp?+simple_rnn_cell_162/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
)simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_162_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_162/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_162_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_162/BiasAddBiasAdd$simple_rnn_cell_162/MatMul:product:02simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_162_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_162/MatMul_1MatMulzeros:output:03simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_162/addAddV2$simple_rnn_cell_162/BiasAdd:output:0&simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_162/TanhTanhsimple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????Pn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_162_matmul_readvariableop_resource3simple_rnn_cell_162_biasadd_readvariableop_resource4simple_rnn_cell_162_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10851884*
condR
while_cond_10851883*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_162/BiasAdd/ReadVariableOp*^simple_rnn_cell_162/MatMul/ReadVariableOp,^simple_rnn_cell_162/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_162/BiasAdd/ReadVariableOp*simple_rnn_cell_162/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_162/MatMul/ReadVariableOp)simple_rnn_cell_162/MatMul/ReadVariableOp2Z
+simple_rnn_cell_162/MatMul_1/ReadVariableOp+simple_rnn_cell_162/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
,__inference_dropout_2_layer_call_fn_10853588

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_10852306s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?

?
6__inference_simple_rnn_cell_162_layer_call_fn_10854155

inputs
states_0
unknown:P
	unknown_0:P
	unknown_1:PP
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????P:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10851420o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Pq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????P
"
_user_specified_name
states/0
?!
?
while_body_10851764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_163_10851786_0:Pd2
$while_simple_rnn_cell_163_10851788_0:d6
$while_simple_rnn_cell_163_10851790_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_163_10851786:Pd0
"while_simple_rnn_cell_163_10851788:d4
"while_simple_rnn_cell_163_10851790:dd??1while/simple_rnn_cell_163/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
1while/simple_rnn_cell_163/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_163_10851786_0$while_simple_rnn_cell_163_10851788_0$while_simple_rnn_cell_163_10851790_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10851712?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_163/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :????
while/Identity_4Identity:while/simple_rnn_cell_163/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp2^while/simple_rnn_cell_163/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_163_10851786$while_simple_rnn_cell_163_10851786_0"J
"while_simple_rnn_cell_163_10851788$while_simple_rnn_cell_163_10851788_0"J
"while_simple_rnn_cell_163_10851790$while_simple_rnn_cell_163_10851790_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2f
1while/simple_rnn_cell_163/StatefulPartitionedCall1while/simple_rnn_cell_163/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?-
?
while_body_10852211
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_163_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_163_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_163_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_163_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_163/MatMul/ReadVariableOp?1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_163_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_163/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_163/BiasAddBiasAdd*while/simple_rnn_cell_163/MatMul:product:08while/simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_163/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_163/addAddV2*while/simple_rnn_cell_163/BiasAdd:output:0,while/simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_163/TanhTanh!while/simple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_163/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_163/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_163/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_163/MatMul/ReadVariableOp2^while/simple_rnn_cell_163/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_163_biasadd_readvariableop_resource;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_163_matmul_1_readvariableop_resource<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_163_matmul_readvariableop_resource:while_simple_rnn_cell_163_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_163/MatMul/ReadVariableOp/while/simple_rnn_cell_163/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_10853403
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_10853403___redundant_placeholder06
2while_while_cond_10853403___redundant_placeholder16
2while_while_cond_10853403___redundant_placeholder26
2while_while_cond_10853403___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?-
?
while_body_10853512
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_162_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_162_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_162_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_162_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_162/MatMul/ReadVariableOp?1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_162_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_162/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_162/BiasAddBiasAdd*while/simple_rnn_cell_162/MatMul:product:08while/simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_162/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_162/addAddV2*while/simple_rnn_cell_162/BiasAdd:output:0,while/simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_162/TanhTanh!while/simple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_162/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_162/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_162/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_162/MatMul/ReadVariableOp2^while/simple_rnn_cell_162/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_162_biasadd_readvariableop_resource;while_simple_rnn_cell_162_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_162_matmul_1_readvariableop_resource<while_simple_rnn_cell_162_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_162_matmul_readvariableop_resource:while_simple_rnn_cell_162_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp0while/simple_rnn_cell_162/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_162/MatMul/ReadVariableOp/while/simple_rnn_cell_162/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp1while/simple_rnn_cell_162/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
ҵ
?

#__inference__wrapped_model_10851252
simple_rnn_2_input^
Lsequential_1_simple_rnn_2_simple_rnn_cell_162_matmul_readvariableop_resource:P[
Msequential_1_simple_rnn_2_simple_rnn_cell_162_biasadd_readvariableop_resource:P`
Nsequential_1_simple_rnn_2_simple_rnn_cell_162_matmul_1_readvariableop_resource:PP^
Lsequential_1_simple_rnn_3_simple_rnn_cell_163_matmul_readvariableop_resource:Pd[
Msequential_1_simple_rnn_3_simple_rnn_cell_163_biasadd_readvariableop_resource:d`
Nsequential_1_simple_rnn_3_simple_rnn_cell_163_matmul_1_readvariableop_resource:ddE
3sequential_1_dense_1_matmul_readvariableop_resource:dB
4sequential_1_dense_1_biasadd_readvariableop_resource:
identity??+sequential_1/dense_1/BiasAdd/ReadVariableOp?*sequential_1/dense_1/MatMul/ReadVariableOp?Dsequential_1/simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOp?Csequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOp?Esequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp?sequential_1/simple_rnn_2/while?Dsequential_1/simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOp?Csequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOp?Esequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp?sequential_1/simple_rnn_3/whilea
sequential_1/simple_rnn_2/ShapeShapesimple_rnn_2_input*
T0*
_output_shapes
:w
-sequential_1/simple_rnn_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential_1/simple_rnn_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential_1/simple_rnn_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'sequential_1/simple_rnn_2/strided_sliceStridedSlice(sequential_1/simple_rnn_2/Shape:output:06sequential_1/simple_rnn_2/strided_slice/stack:output:08sequential_1/simple_rnn_2/strided_slice/stack_1:output:08sequential_1/simple_rnn_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_1/simple_rnn_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
&sequential_1/simple_rnn_2/zeros/packedPack0sequential_1/simple_rnn_2/strided_slice:output:01sequential_1/simple_rnn_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%sequential_1/simple_rnn_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_1/simple_rnn_2/zerosFill/sequential_1/simple_rnn_2/zeros/packed:output:0.sequential_1/simple_rnn_2/zeros/Const:output:0*
T0*'
_output_shapes
:?????????P}
(sequential_1/simple_rnn_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
#sequential_1/simple_rnn_2/transpose	Transposesimple_rnn_2_input1sequential_1/simple_rnn_2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????x
!sequential_1/simple_rnn_2/Shape_1Shape'sequential_1/simple_rnn_2/transpose:y:0*
T0*
_output_shapes
:y
/sequential_1/simple_rnn_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_1/simple_rnn_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_1/simple_rnn_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_1/simple_rnn_2/strided_slice_1StridedSlice*sequential_1/simple_rnn_2/Shape_1:output:08sequential_1/simple_rnn_2/strided_slice_1/stack:output:0:sequential_1/simple_rnn_2/strided_slice_1/stack_1:output:0:sequential_1/simple_rnn_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
5sequential_1/simple_rnn_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'sequential_1/simple_rnn_2/TensorArrayV2TensorListReserve>sequential_1/simple_rnn_2/TensorArrayV2/element_shape:output:02sequential_1/simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Osequential_1/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Asequential_1/simple_rnn_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_1/simple_rnn_2/transpose:y:0Xsequential_1/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???y
/sequential_1/simple_rnn_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_1/simple_rnn_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_1/simple_rnn_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_1/simple_rnn_2/strided_slice_2StridedSlice'sequential_1/simple_rnn_2/transpose:y:08sequential_1/simple_rnn_2/strided_slice_2/stack:output:0:sequential_1/simple_rnn_2/strided_slice_2/stack_1:output:0:sequential_1/simple_rnn_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
Csequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOpReadVariableOpLsequential_1_simple_rnn_2_simple_rnn_cell_162_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
4sequential_1/simple_rnn_2/simple_rnn_cell_162/MatMulMatMul2sequential_1/simple_rnn_2/strided_slice_2:output:0Ksequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Dsequential_1/simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOpReadVariableOpMsequential_1_simple_rnn_2_simple_rnn_cell_162_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
5sequential_1/simple_rnn_2/simple_rnn_cell_162/BiasAddBiasAdd>sequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul:product:0Lsequential_1/simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Esequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOpReadVariableOpNsequential_1_simple_rnn_2_simple_rnn_cell_162_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
6sequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul_1MatMul(sequential_1/simple_rnn_2/zeros:output:0Msequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1sequential_1/simple_rnn_2/simple_rnn_cell_162/addAddV2>sequential_1/simple_rnn_2/simple_rnn_cell_162/BiasAdd:output:0@sequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
2sequential_1/simple_rnn_2/simple_rnn_cell_162/TanhTanh5sequential_1/simple_rnn_2/simple_rnn_cell_162/add:z:0*
T0*'
_output_shapes
:?????????P?
7sequential_1/simple_rnn_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)sequential_1/simple_rnn_2/TensorArrayV2_1TensorListReserve@sequential_1/simple_rnn_2/TensorArrayV2_1/element_shape:output:02sequential_1/simple_rnn_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???`
sequential_1/simple_rnn_2/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2sequential_1/simple_rnn_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????n
,sequential_1/simple_rnn_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_1/simple_rnn_2/whileWhile5sequential_1/simple_rnn_2/while/loop_counter:output:0;sequential_1/simple_rnn_2/while/maximum_iterations:output:0'sequential_1/simple_rnn_2/time:output:02sequential_1/simple_rnn_2/TensorArrayV2_1:handle:0(sequential_1/simple_rnn_2/zeros:output:02sequential_1/simple_rnn_2/strided_slice_1:output:0Qsequential_1/simple_rnn_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0Lsequential_1_simple_rnn_2_simple_rnn_cell_162_matmul_readvariableop_resourceMsequential_1_simple_rnn_2_simple_rnn_cell_162_biasadd_readvariableop_resourceNsequential_1_simple_rnn_2_simple_rnn_cell_162_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *9
body1R/
-sequential_1_simple_rnn_2_while_body_10851074*9
cond1R/
-sequential_1_simple_rnn_2_while_cond_10851073*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
Jsequential_1/simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
<sequential_1/simple_rnn_2/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_1/simple_rnn_2/while:output:3Ssequential_1/simple_rnn_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0?
/sequential_1/simple_rnn_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1sequential_1/simple_rnn_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1sequential_1/simple_rnn_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_1/simple_rnn_2/strided_slice_3StridedSliceEsequential_1/simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:08sequential_1/simple_rnn_2/strided_slice_3/stack:output:0:sequential_1/simple_rnn_2/strided_slice_3/stack_1:output:0:sequential_1/simple_rnn_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask
*sequential_1/simple_rnn_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
%sequential_1/simple_rnn_2/transpose_1	TransposeEsequential_1/simple_rnn_2/TensorArrayV2Stack/TensorListStack:tensor:03sequential_1/simple_rnn_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P?
sequential_1/dropout_2/IdentityIdentity)sequential_1/simple_rnn_2/transpose_1:y:0*
T0*+
_output_shapes
:?????????Pw
sequential_1/simple_rnn_3/ShapeShape(sequential_1/dropout_2/Identity:output:0*
T0*
_output_shapes
:w
-sequential_1/simple_rnn_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential_1/simple_rnn_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential_1/simple_rnn_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'sequential_1/simple_rnn_3/strided_sliceStridedSlice(sequential_1/simple_rnn_3/Shape:output:06sequential_1/simple_rnn_3/strided_slice/stack:output:08sequential_1/simple_rnn_3/strided_slice/stack_1:output:08sequential_1/simple_rnn_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_1/simple_rnn_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
&sequential_1/simple_rnn_3/zeros/packedPack0sequential_1/simple_rnn_3/strided_slice:output:01sequential_1/simple_rnn_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%sequential_1/simple_rnn_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_1/simple_rnn_3/zerosFill/sequential_1/simple_rnn_3/zeros/packed:output:0.sequential_1/simple_rnn_3/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d}
(sequential_1/simple_rnn_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
#sequential_1/simple_rnn_3/transpose	Transpose(sequential_1/dropout_2/Identity:output:01sequential_1/simple_rnn_3/transpose/perm:output:0*
T0*+
_output_shapes
:?????????Px
!sequential_1/simple_rnn_3/Shape_1Shape'sequential_1/simple_rnn_3/transpose:y:0*
T0*
_output_shapes
:y
/sequential_1/simple_rnn_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_1/simple_rnn_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_1/simple_rnn_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_1/simple_rnn_3/strided_slice_1StridedSlice*sequential_1/simple_rnn_3/Shape_1:output:08sequential_1/simple_rnn_3/strided_slice_1/stack:output:0:sequential_1/simple_rnn_3/strided_slice_1/stack_1:output:0:sequential_1/simple_rnn_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
5sequential_1/simple_rnn_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'sequential_1/simple_rnn_3/TensorArrayV2TensorListReserve>sequential_1/simple_rnn_3/TensorArrayV2/element_shape:output:02sequential_1/simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Osequential_1/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
Asequential_1/simple_rnn_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_1/simple_rnn_3/transpose:y:0Xsequential_1/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???y
/sequential_1/simple_rnn_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_1/simple_rnn_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_1/simple_rnn_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_1/simple_rnn_3/strided_slice_2StridedSlice'sequential_1/simple_rnn_3/transpose:y:08sequential_1/simple_rnn_3/strided_slice_2/stack:output:0:sequential_1/simple_rnn_3/strided_slice_2/stack_1:output:0:sequential_1/simple_rnn_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
Csequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOpLsequential_1_simple_rnn_3_simple_rnn_cell_163_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
4sequential_1/simple_rnn_3/simple_rnn_cell_163/MatMulMatMul2sequential_1/simple_rnn_3/strided_slice_2:output:0Ksequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Dsequential_1/simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOpMsequential_1_simple_rnn_3_simple_rnn_cell_163_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
5sequential_1/simple_rnn_3/simple_rnn_cell_163/BiasAddBiasAdd>sequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul:product:0Lsequential_1/simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Esequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOpNsequential_1_simple_rnn_3_simple_rnn_cell_163_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
6sequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul_1MatMul(sequential_1/simple_rnn_3/zeros:output:0Msequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1sequential_1/simple_rnn_3/simple_rnn_cell_163/addAddV2>sequential_1/simple_rnn_3/simple_rnn_cell_163/BiasAdd:output:0@sequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
2sequential_1/simple_rnn_3/simple_rnn_cell_163/TanhTanh5sequential_1/simple_rnn_3/simple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????d?
7sequential_1/simple_rnn_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
)sequential_1/simple_rnn_3/TensorArrayV2_1TensorListReserve@sequential_1/simple_rnn_3/TensorArrayV2_1/element_shape:output:02sequential_1/simple_rnn_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???`
sequential_1/simple_rnn_3/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2sequential_1/simple_rnn_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????n
,sequential_1/simple_rnn_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_1/simple_rnn_3/whileWhile5sequential_1/simple_rnn_3/while/loop_counter:output:0;sequential_1/simple_rnn_3/while/maximum_iterations:output:0'sequential_1/simple_rnn_3/time:output:02sequential_1/simple_rnn_3/TensorArrayV2_1:handle:0(sequential_1/simple_rnn_3/zeros:output:02sequential_1/simple_rnn_3/strided_slice_1:output:0Qsequential_1/simple_rnn_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0Lsequential_1_simple_rnn_3_simple_rnn_cell_163_matmul_readvariableop_resourceMsequential_1_simple_rnn_3_simple_rnn_cell_163_biasadd_readvariableop_resourceNsequential_1_simple_rnn_3_simple_rnn_cell_163_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *9
body1R/
-sequential_1_simple_rnn_3_while_body_10851179*9
cond1R/
-sequential_1_simple_rnn_3_while_cond_10851178*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
Jsequential_1/simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
<sequential_1/simple_rnn_3/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_1/simple_rnn_3/while:output:3Ssequential_1/simple_rnn_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0?
/sequential_1/simple_rnn_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1sequential_1/simple_rnn_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1sequential_1/simple_rnn_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_1/simple_rnn_3/strided_slice_3StridedSliceEsequential_1/simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:08sequential_1/simple_rnn_3/strided_slice_3/stack:output:0:sequential_1/simple_rnn_3/strided_slice_3/stack_1:output:0:sequential_1/simple_rnn_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask
*sequential_1/simple_rnn_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
%sequential_1/simple_rnn_3/transpose_1	TransposeEsequential_1/simple_rnn_3/TensorArrayV2Stack/TensorListStack:tensor:03sequential_1/simple_rnn_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d?
sequential_1/dropout_3/IdentityIdentity2sequential_1/simple_rnn_3/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
sequential_1/dense_1/MatMulMatMul(sequential_1/dropout_3/Identity:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_1/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOpE^sequential_1/simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOpD^sequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOpF^sequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp ^sequential_1/simple_rnn_2/whileE^sequential_1/simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOpD^sequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOpF^sequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp ^sequential_1/simple_rnn_3/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2?
Dsequential_1/simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOpDsequential_1/simple_rnn_2/simple_rnn_cell_162/BiasAdd/ReadVariableOp2?
Csequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOpCsequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul/ReadVariableOp2?
Esequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOpEsequential_1/simple_rnn_2/simple_rnn_cell_162/MatMul_1/ReadVariableOp2B
sequential_1/simple_rnn_2/whilesequential_1/simple_rnn_2/while2?
Dsequential_1/simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOpDsequential_1/simple_rnn_3/simple_rnn_cell_163/BiasAdd/ReadVariableOp2?
Csequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOpCsequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul/ReadVariableOp2?
Esequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOpEsequential_1/simple_rnn_3/simple_rnn_cell_163/MatMul_1/ReadVariableOp2B
sequential_1/simple_rnn_3/whilesequential_1/simple_rnn_3/while:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_2_input
?
?
-sequential_1_simple_rnn_2_while_cond_10851073P
Lsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_loop_counterV
Rsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_maximum_iterations/
+sequential_1_simple_rnn_2_while_placeholder1
-sequential_1_simple_rnn_2_while_placeholder_11
-sequential_1_simple_rnn_2_while_placeholder_2R
Nsequential_1_simple_rnn_2_while_less_sequential_1_simple_rnn_2_strided_slice_1j
fsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_cond_10851073___redundant_placeholder0j
fsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_cond_10851073___redundant_placeholder1j
fsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_cond_10851073___redundant_placeholder2j
fsequential_1_simple_rnn_2_while_sequential_1_simple_rnn_2_while_cond_10851073___redundant_placeholder3,
(sequential_1_simple_rnn_2_while_identity
?
$sequential_1/simple_rnn_2/while/LessLess+sequential_1_simple_rnn_2_while_placeholderNsequential_1_simple_rnn_2_while_less_sequential_1_simple_rnn_2_strided_slice_1*
T0*
_output_shapes
: 
(sequential_1/simple_rnn_2/while/IdentityIdentity(sequential_1/simple_rnn_2/while/Less:z:0*
T0
*
_output_shapes
: "]
(sequential_1_simple_rnn_2_while_identity1sequential_1/simple_rnn_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?
?
/__inference_simple_rnn_2_layer_call_fn_10853113
inputs_0
unknown:P
	unknown_0:P
	unknown_1:PP
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10851376|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?=
?
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10854081

inputsD
2simple_rnn_cell_163_matmul_readvariableop_resource:PdA
3simple_rnn_cell_163_biasadd_readvariableop_resource:dF
4simple_rnn_cell_163_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_163/BiasAdd/ReadVariableOp?)simple_rnn_cell_163/MatMul/ReadVariableOp?+simple_rnn_cell_163/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
)simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_163_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_163/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_163_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_163/BiasAddBiasAdd$simple_rnn_cell_163/MatMul:product:02simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_163_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_163/MatMul_1MatMulzeros:output:03simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_163/addAddV2$simple_rnn_cell_163/BiasAdd:output:0&simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_163/TanhTanhsimple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_163_matmul_readvariableop_resource3simple_rnn_cell_163_biasadd_readvariableop_resource4simple_rnn_cell_163_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_10854015*
condR
while_cond_10854014*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_163/BiasAdd/ReadVariableOp*^simple_rnn_cell_163/MatMul/ReadVariableOp,^simple_rnn_cell_163/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_163/BiasAdd/ReadVariableOp*simple_rnn_cell_163/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_163/MatMul/ReadVariableOp)simple_rnn_cell_163/MatMul/ReadVariableOp2Z
+simple_rnn_cell_163/MatMul_1/ReadVariableOp+simple_rnn_cell_163/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?!
?
while_body_10851472
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_162_10851494_0:P2
$while_simple_rnn_cell_162_10851496_0:P6
$while_simple_rnn_cell_162_10851498_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_162_10851494:P0
"while_simple_rnn_cell_162_10851496:P4
"while_simple_rnn_cell_162_10851498:PP??1while/simple_rnn_cell_162/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/simple_rnn_cell_162/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_162_10851494_0$while_simple_rnn_cell_162_10851496_0$while_simple_rnn_cell_162_10851498_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????P:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10851420?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_162/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :????
while/Identity_4Identity:while/simple_rnn_cell_162/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp2^while/simple_rnn_cell_162/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_162_10851494$while_simple_rnn_cell_162_10851494_0"J
"while_simple_rnn_cell_162_10851496$while_simple_rnn_cell_162_10851496_0"J
"while_simple_rnn_cell_162_10851498$while_simple_rnn_cell_162_10851498_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2f
1while/simple_rnn_cell_162/StatefulPartitionedCall1while/simple_rnn_cell_162/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?
?
/__inference_simple_rnn_2_layer_call_fn_10853146

inputs
unknown:P
	unknown_0:P
	unknown_1:PP
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10852430s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
while_body_10854015
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_163_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_163_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_163_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_163_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_163/MatMul/ReadVariableOp?1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_163/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_163_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_163/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_163/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_163/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_163/BiasAddBiasAdd*while/simple_rnn_cell_163/MatMul:product:08while/simple_rnn_cell_163/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_163/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_163/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_163/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_163/addAddV2*while/simple_rnn_cell_163/BiasAdd:output:0,while/simple_rnn_cell_163/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_163/TanhTanh!while/simple_rnn_cell_163/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_163/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_163/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_163/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_163/MatMul/ReadVariableOp2^while/simple_rnn_cell_163/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_163_biasadd_readvariableop_resource;while_simple_rnn_cell_163_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_163_matmul_1_readvariableop_resource<while_simple_rnn_cell_163_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_163_matmul_readvariableop_resource:while_simple_rnn_cell_163_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp0while/simple_rnn_cell_163/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_163/MatMul/ReadVariableOp/while/simple_rnn_cell_163/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp1while/simple_rnn_cell_163/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?	
?
&__inference_signature_wrapper_10853102
simple_rnn_2_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_10851252o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_2_input
?
?
*__inference_dense_1_layer_call_fn_10854117

inputs
unknown:d
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_10852097o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?
$__inference__traced_restore_10854470
file_prefix1
assignvariableop_dense_1_kernel:d-
assignvariableop_1_dense_1_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: J
8assignvariableop_7_simple_rnn_2_simple_rnn_cell_2_kernel:PT
Bassignvariableop_8_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel:PPD
6assignvariableop_9_simple_rnn_2_simple_rnn_cell_2_bias:PK
9assignvariableop_10_simple_rnn_3_simple_rnn_cell_3_kernel:PdU
Cassignvariableop_11_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel:ddE
7assignvariableop_12_simple_rnn_3_simple_rnn_cell_3_bias:d#
assignvariableop_13_total: #
assignvariableop_14_count: ;
)assignvariableop_15_adam_dense_1_kernel_m:d5
'assignvariableop_16_adam_dense_1_bias_m:R
@assignvariableop_17_adam_simple_rnn_2_simple_rnn_cell_2_kernel_m:P\
Jassignvariableop_18_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_m:PPL
>assignvariableop_19_adam_simple_rnn_2_simple_rnn_cell_2_bias_m:PR
@assignvariableop_20_adam_simple_rnn_3_simple_rnn_cell_3_kernel_m:Pd\
Jassignvariableop_21_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_m:ddL
>assignvariableop_22_adam_simple_rnn_3_simple_rnn_cell_3_bias_m:d;
)assignvariableop_23_adam_dense_1_kernel_v:d5
'assignvariableop_24_adam_dense_1_bias_v:R
@assignvariableop_25_adam_simple_rnn_2_simple_rnn_cell_2_kernel_v:P\
Jassignvariableop_26_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_v:PPL
>assignvariableop_27_adam_simple_rnn_2_simple_rnn_cell_2_bias_v:PR
@assignvariableop_28_adam_simple_rnn_3_simple_rnn_cell_3_kernel_v:Pd\
Jassignvariableop_29_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_v:ddL
>assignvariableop_30_adam_simple_rnn_3_simple_rnn_cell_3_bias_v:d
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp8assignvariableop_7_simple_rnn_2_simple_rnn_cell_2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpBassignvariableop_8_simple_rnn_2_simple_rnn_cell_2_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp6assignvariableop_9_simple_rnn_2_simple_rnn_cell_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp9assignvariableop_10_simple_rnn_3_simple_rnn_cell_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpCassignvariableop_11_simple_rnn_3_simple_rnn_cell_3_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_simple_rnn_3_simple_rnn_cell_3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp@assignvariableop_17_adam_simple_rnn_2_simple_rnn_cell_2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpJassignvariableop_18_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp>assignvariableop_19_adam_simple_rnn_2_simple_rnn_cell_2_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_simple_rnn_3_simple_rnn_cell_3_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpJassignvariableop_21_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp>assignvariableop_22_adam_simple_rnn_3_simple_rnn_cell_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp@assignvariableop_25_adam_simple_rnn_2_simple_rnn_cell_2_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpJassignvariableop_26_adam_simple_rnn_2_simple_rnn_cell_2_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp>assignvariableop_27_adam_simple_rnn_2_simple_rnn_cell_2_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp@assignvariableop_28_adam_simple_rnn_3_simple_rnn_cell_3_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpJassignvariableop_29_adam_simple_rnn_3_simple_rnn_cell_3_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp>assignvariableop_30_adam_simple_rnn_3_simple_rnn_cell_3_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
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
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
simple_rnn_2_input?
$serving_default_simple_rnn_2_input:0?????????;
dense_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer

signatures
#_self_saveable_object_factories
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_sequential
?
cell

state_spec
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
?
!cell
"
state_spec
##_self_saveable_object_factories
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
#*_self_saveable_object_factories
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
?

2kernel
3bias
#4_self_saveable_object_factories
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
?
;iter

<beta_1

=beta_2
	>decay
?learning_rate2m?3m?Am?Bm?Cm?Dm?Em?Fm?2v?3v?Av?Bv?Cv?Dv?Ev?Fv?"
	optimizer
,
@serving_default"
signature_map
 "
trackable_dict_wrapper
X
A0
B1
C2
D3
E4
F5
26
37"
trackable_list_wrapper
X
A0
B1
C2
D3
E4
F5
26
37"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_sequential_1_layer_call_fn_10852123
/__inference_sequential_1_layer_call_fn_10852604
/__inference_sequential_1_layer_call_fn_10852625
/__inference_sequential_1_layer_call_fn_10852527?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852845
J__inference_sequential_1_layer_call_and_return_conditional_losses_10853079
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852552
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852577?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference__wrapped_model_10851252simple_rnn_2_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?

Akernel
Brecurrent_kernel
Cbias
#L_self_saveable_object_factories
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q_random_generator
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
A0
B1
C2"
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Tstates
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_simple_rnn_2_layer_call_fn_10853113
/__inference_simple_rnn_2_layer_call_fn_10853124
/__inference_simple_rnn_2_layer_call_fn_10853135
/__inference_simple_rnn_2_layer_call_fn_10853146?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10853254
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10853362
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10853470
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10853578?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
,__inference_dropout_2_layer_call_fn_10853583
,__inference_dropout_2_layer_call_fn_10853588?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_2_layer_call_and_return_conditional_losses_10853593
G__inference_dropout_2_layer_call_and_return_conditional_losses_10853605?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?

Dkernel
Erecurrent_kernel
Fbias
#__self_saveable_object_factories
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d_random_generator
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
D0
E1
F2"
trackable_list_wrapper
5
D0
E1
F2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

gstates
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_simple_rnn_3_layer_call_fn_10853616
/__inference_simple_rnn_3_layer_call_fn_10853627
/__inference_simple_rnn_3_layer_call_fn_10853638
/__inference_simple_rnn_3_layer_call_fn_10853649?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10853757
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10853865
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10853973
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10854081?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
,__inference_dropout_3_layer_call_fn_10854086
,__inference_dropout_3_layer_call_fn_10854091?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_3_layer_call_and_return_conditional_losses_10854096
G__inference_dropout_3_layer_call_and_return_conditional_losses_10854108?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 :d2dense_1/kernel
:2dense_1/bias
 "
trackable_dict_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_1_layer_call_fn_10854117?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_1_layer_call_and_return_conditional_losses_10854127?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
&__inference_signature_wrapper_10853102simple_rnn_2_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
7:5P2%simple_rnn_2/simple_rnn_cell_2/kernel
A:?PP2/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel
1:/P2#simple_rnn_2/simple_rnn_cell_2/bias
7:5Pd2%simple_rnn_3/simple_rnn_cell_3/kernel
A:?dd2/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel
1:/d2#simple_rnn_3/simple_rnn_cell_3/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
w0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
5
A0
B1
C2"
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
6__inference_simple_rnn_cell_162_layer_call_fn_10854141
6__inference_simple_rnn_cell_162_layer_call_fn_10854155?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10854172
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10854189?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
trackable_dict_wrapper
5
D0
E1
F2"
trackable_list_wrapper
5
D0
E1
F2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
6__inference_simple_rnn_cell_163_layer_call_fn_10854203
6__inference_simple_rnn_cell_163_layer_call_fn_10854217?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10854234
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10854251?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
!0"
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

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
%:#d2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
<::P2,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/m
F:DPP26Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/m
6:4P2*Adam/simple_rnn_2/simple_rnn_cell_2/bias/m
<::Pd2,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/m
F:Ddd26Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/m
6:4d2*Adam/simple_rnn_3/simple_rnn_cell_3/bias/m
%:#d2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
<::P2,Adam/simple_rnn_2/simple_rnn_cell_2/kernel/v
F:DPP26Adam/simple_rnn_2/simple_rnn_cell_2/recurrent_kernel/v
6:4P2*Adam/simple_rnn_2/simple_rnn_cell_2/bias/v
<::Pd2,Adam/simple_rnn_3/simple_rnn_cell_3/kernel/v
F:Ddd26Adam/simple_rnn_3/simple_rnn_cell_3/recurrent_kernel/v
6:4d2*Adam/simple_rnn_3/simple_rnn_cell_3/bias/v?
#__inference__wrapped_model_10851252~ACBDFE23??<
5?2
0?-
simple_rnn_2_input?????????
? "1?.
,
dense_1!?
dense_1??????????
E__inference_dense_1_layer_call_and_return_conditional_losses_10854127\23/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? }
*__inference_dense_1_layer_call_fn_10854117O23/?,
%?"
 ?
inputs?????????d
? "???????????
G__inference_dropout_2_layer_call_and_return_conditional_losses_10853593d7?4
-?*
$?!
inputs?????????P
p 
? ")?&
?
0?????????P
? ?
G__inference_dropout_2_layer_call_and_return_conditional_losses_10853605d7?4
-?*
$?!
inputs?????????P
p
? ")?&
?
0?????????P
? ?
,__inference_dropout_2_layer_call_fn_10853583W7?4
-?*
$?!
inputs?????????P
p 
? "??????????P?
,__inference_dropout_2_layer_call_fn_10853588W7?4
-?*
$?!
inputs?????????P
p
? "??????????P?
G__inference_dropout_3_layer_call_and_return_conditional_losses_10854096\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? ?
G__inference_dropout_3_layer_call_and_return_conditional_losses_10854108\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? 
,__inference_dropout_3_layer_call_fn_10854086O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d
,__inference_dropout_3_layer_call_fn_10854091O3?0
)?&
 ?
inputs?????????d
p
? "??????????d?
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852552zACBDFE23G?D
=?:
0?-
simple_rnn_2_input?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852577zACBDFE23G?D
=?:
0?-
simple_rnn_2_input?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_10852845nACBDFE23;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_1_layer_call_and_return_conditional_losses_10853079nACBDFE23;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_1_layer_call_fn_10852123mACBDFE23G?D
=?:
0?-
simple_rnn_2_input?????????
p 

 
? "???????????
/__inference_sequential_1_layer_call_fn_10852527mACBDFE23G?D
=?:
0?-
simple_rnn_2_input?????????
p

 
? "???????????
/__inference_sequential_1_layer_call_fn_10852604aACBDFE23;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
/__inference_sequential_1_layer_call_fn_10852625aACBDFE23;?8
1?.
$?!
inputs?????????
p

 
? "???????????
&__inference_signature_wrapper_10853102?ACBDFE23U?R
? 
K?H
F
simple_rnn_2_input0?-
simple_rnn_2_input?????????"1?.
,
dense_1!?
dense_1??????????
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10853254?ACBO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "2?/
(?%
0??????????????????P
? ?
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10853362?ACBO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "2?/
(?%
0??????????????????P
? ?
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10853470qACB??<
5?2
$?!
inputs?????????

 
p 

 
? ")?&
?
0?????????P
? ?
J__inference_simple_rnn_2_layer_call_and_return_conditional_losses_10853578qACB??<
5?2
$?!
inputs?????????

 
p

 
? ")?&
?
0?????????P
? ?
/__inference_simple_rnn_2_layer_call_fn_10853113}ACBO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"??????????????????P?
/__inference_simple_rnn_2_layer_call_fn_10853124}ACBO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"??????????????????P?
/__inference_simple_rnn_2_layer_call_fn_10853135dACB??<
5?2
$?!
inputs?????????

 
p 

 
? "??????????P?
/__inference_simple_rnn_2_layer_call_fn_10853146dACB??<
5?2
$?!
inputs?????????

 
p

 
? "??????????P?
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10853757}DFEO?L
E?B
4?1
/?,
inputs/0??????????????????P

 
p 

 
? "%?"
?
0?????????d
? ?
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10853865}DFEO?L
E?B
4?1
/?,
inputs/0??????????????????P

 
p

 
? "%?"
?
0?????????d
? ?
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10853973mDFE??<
5?2
$?!
inputs?????????P

 
p 

 
? "%?"
?
0?????????d
? ?
J__inference_simple_rnn_3_layer_call_and_return_conditional_losses_10854081mDFE??<
5?2
$?!
inputs?????????P

 
p

 
? "%?"
?
0?????????d
? ?
/__inference_simple_rnn_3_layer_call_fn_10853616pDFEO?L
E?B
4?1
/?,
inputs/0??????????????????P

 
p 

 
? "??????????d?
/__inference_simple_rnn_3_layer_call_fn_10853627pDFEO?L
E?B
4?1
/?,
inputs/0??????????????????P

 
p

 
? "??????????d?
/__inference_simple_rnn_3_layer_call_fn_10853638`DFE??<
5?2
$?!
inputs?????????P

 
p 

 
? "??????????d?
/__inference_simple_rnn_3_layer_call_fn_10853649`DFE??<
5?2
$?!
inputs?????????P

 
p

 
? "??????????d?
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10854172?ACB\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????P
p 
? "R?O
H?E
?
0/0?????????P
$?!
?
0/1/0?????????P
? ?
Q__inference_simple_rnn_cell_162_layer_call_and_return_conditional_losses_10854189?ACB\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????P
p
? "R?O
H?E
?
0/0?????????P
$?!
?
0/1/0?????????P
? ?
6__inference_simple_rnn_cell_162_layer_call_fn_10854141?ACB\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????P
p 
? "D?A
?
0?????????P
"?
?
1/0?????????P?
6__inference_simple_rnn_cell_162_layer_call_fn_10854155?ACB\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????P
p
? "D?A
?
0?????????P
"?
?
1/0?????????P?
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10854234?DFE\?Y
R?O
 ?
inputs?????????P
'?$
"?
states/0?????????d
p 
? "R?O
H?E
?
0/0?????????d
$?!
?
0/1/0?????????d
? ?
Q__inference_simple_rnn_cell_163_layer_call_and_return_conditional_losses_10854251?DFE\?Y
R?O
 ?
inputs?????????P
'?$
"?
states/0?????????d
p
? "R?O
H?E
?
0/0?????????d
$?!
?
0/1/0?????????d
? ?
6__inference_simple_rnn_cell_163_layer_call_fn_10854203?DFE\?Y
R?O
 ?
inputs?????????P
'?$
"?
states/0?????????d
p 
? "D?A
?
0?????????d
"?
?
1/0?????????d?
6__inference_simple_rnn_cell_163_layer_call_fn_10854217?DFE\?Y
R?O
 ?
inputs?????????P
'?$
"?
states/0?????????d
p
? "D?A
?
0?????????d
"?
?
1/0?????????d