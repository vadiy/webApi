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
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:d*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
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
%simple_rnn_6/simple_rnn_cell_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*6
shared_name'%simple_rnn_6/simple_rnn_cell_6/kernel
?
9simple_rnn_6/simple_rnn_cell_6/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_6/simple_rnn_cell_6/kernel*
_output_shapes

:P*
dtype0
?
/simple_rnn_6/simple_rnn_cell_6/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*@
shared_name1/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel
?
Csimple_rnn_6/simple_rnn_cell_6/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel*
_output_shapes

:PP*
dtype0
?
#simple_rnn_6/simple_rnn_cell_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*4
shared_name%#simple_rnn_6/simple_rnn_cell_6/bias
?
7simple_rnn_6/simple_rnn_cell_6/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_6/simple_rnn_cell_6/bias*
_output_shapes
:P*
dtype0
?
%simple_rnn_7/simple_rnn_cell_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*6
shared_name'%simple_rnn_7/simple_rnn_cell_7/kernel
?
9simple_rnn_7/simple_rnn_cell_7/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_7/simple_rnn_cell_7/kernel*
_output_shapes

:Pd*
dtype0
?
/simple_rnn_7/simple_rnn_cell_7/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*@
shared_name1/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel
?
Csimple_rnn_7/simple_rnn_cell_7/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel*
_output_shapes

:dd*
dtype0
?
#simple_rnn_7/simple_rnn_cell_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#simple_rnn_7/simple_rnn_cell_7/bias
?
7simple_rnn_7/simple_rnn_cell_7/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_7/simple_rnn_cell_7/bias*
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
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*=
shared_name.,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m
?
@Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m*
_output_shapes

:P*
dtype0
?
6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*G
shared_name86Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m
?
JAdam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m*
_output_shapes

:PP*
dtype0
?
*Adam/simple_rnn_6/simple_rnn_cell_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*;
shared_name,*Adam/simple_rnn_6/simple_rnn_cell_6/bias/m
?
>Adam/simple_rnn_6/simple_rnn_cell_6/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_6/simple_rnn_cell_6/bias/m*
_output_shapes
:P*
dtype0
?
,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*=
shared_name.,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/m
?
@Adam/simple_rnn_7/simple_rnn_cell_7/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/m*
_output_shapes

:Pd*
dtype0
?
6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*G
shared_name86Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/m
?
JAdam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/m*
_output_shapes

:dd*
dtype0
?
*Adam/simple_rnn_7/simple_rnn_cell_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*;
shared_name,*Adam/simple_rnn_7/simple_rnn_cell_7/bias/m
?
>Adam/simple_rnn_7/simple_rnn_cell_7/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_7/simple_rnn_cell_7/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
?
,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*=
shared_name.,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v
?
@Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v*
_output_shapes

:P*
dtype0
?
6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*G
shared_name86Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v
?
JAdam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v*
_output_shapes

:PP*
dtype0
?
*Adam/simple_rnn_6/simple_rnn_cell_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*;
shared_name,*Adam/simple_rnn_6/simple_rnn_cell_6/bias/v
?
>Adam/simple_rnn_6/simple_rnn_cell_6/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_6/simple_rnn_cell_6/bias/v*
_output_shapes
:P*
dtype0
?
,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*=
shared_name.,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/v
?
@Adam/simple_rnn_7/simple_rnn_cell_7/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/v*
_output_shapes

:Pd*
dtype0
?
6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*G
shared_name86Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/v
?
JAdam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/v*
_output_shapes

:dd*
dtype0
?
*Adam/simple_rnn_7/simple_rnn_cell_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*;
shared_name,*Adam/simple_rnn_7/simple_rnn_cell_7/bias/v
?
>Adam/simple_rnn_7/simple_rnn_cell_7/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_7/simple_rnn_cell_7/bias/v*
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
VARIABLE_VALUEdense_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUE%simple_rnn_6/simple_rnn_cell_6/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_6/simple_rnn_cell_6/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_7/simple_rnn_cell_7/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_7/simple_rnn_cell_7/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/simple_rnn_6/simple_rnn_cell_6/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/simple_rnn_7/simple_rnn_cell_7/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/simple_rnn_6/simple_rnn_cell_6/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/simple_rnn_7/simple_rnn_cell_7/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
"serving_default_simple_rnn_6_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall"serving_default_simple_rnn_6_input%simple_rnn_6/simple_rnn_cell_6/kernel#simple_rnn_6/simple_rnn_cell_6/bias/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel%simple_rnn_7/simple_rnn_cell_7/kernel#simple_rnn_7/simple_rnn_cell_7/bias/simple_rnn_7/simple_rnn_cell_7/recurrent_kerneldense_3/kerneldense_3/bias*
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
&__inference_signature_wrapper_11019488
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9simple_rnn_6/simple_rnn_cell_6/kernel/Read/ReadVariableOpCsimple_rnn_6/simple_rnn_cell_6/recurrent_kernel/Read/ReadVariableOp7simple_rnn_6/simple_rnn_cell_6/bias/Read/ReadVariableOp9simple_rnn_7/simple_rnn_cell_7/kernel/Read/ReadVariableOpCsimple_rnn_7/simple_rnn_cell_7/recurrent_kernel/Read/ReadVariableOp7simple_rnn_7/simple_rnn_cell_7/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp@Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_6/simple_rnn_cell_6/bias/m/Read/ReadVariableOp@Adam/simple_rnn_7/simple_rnn_cell_7/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_7/simple_rnn_cell_7/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp@Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_6/simple_rnn_cell_6/bias/v/Read/ReadVariableOp@Adam/simple_rnn_7/simple_rnn_cell_7/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_7/simple_rnn_cell_7/bias/v/Read/ReadVariableOpConst*,
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
!__inference__traced_save_11020753
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%simple_rnn_6/simple_rnn_cell_6/kernel/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel#simple_rnn_6/simple_rnn_cell_6/bias%simple_rnn_7/simple_rnn_cell_7/kernel/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel#simple_rnn_7/simple_rnn_cell_7/biastotalcountAdam/dense_3/kernel/mAdam/dense_3/bias/m,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m*Adam/simple_rnn_6/simple_rnn_cell_6/bias/m,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/m6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/m*Adam/simple_rnn_7/simple_rnn_cell_7/bias/mAdam/dense_3/kernel/vAdam/dense_3/bias/v,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v*Adam/simple_rnn_6/simple_rnn_cell_6/bias/v,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/v6Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/v*Adam/simple_rnn_7/simple_rnn_cell_7/bias/v*+
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
$__inference__traced_restore_11020856??
?
e
G__inference_dropout_7_layer_call_and_return_conditional_losses_11020482

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
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_11018873

inputs'
simple_rnn_6_11018851:P#
simple_rnn_6_11018853:P'
simple_rnn_6_11018855:PP'
simple_rnn_7_11018859:Pd#
simple_rnn_7_11018861:d'
simple_rnn_7_11018863:dd"
dense_3_11018867:d
dense_3_11018869:
identity??dense_3/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?$simple_rnn_6/StatefulPartitionedCall?$simple_rnn_7/StatefulPartitionedCall?
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_6_11018851simple_rnn_6_11018853simple_rnn_6_11018855*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11018816?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_6_layer_call_and_return_conditional_losses_11018692?
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0simple_rnn_7_11018859simple_rnn_7_11018861simple_rnn_7_11018863*
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
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11018663?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_7/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
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
G__inference_dropout_7_layer_call_and_return_conditional_losses_11018539?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_3_11018867dense_3_11018869*
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
E__inference_dense_3_layer_call_and_return_conditional_losses_11018483w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_3/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?=
?
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11018336

inputsD
2simple_rnn_cell_166_matmul_readvariableop_resource:PA
3simple_rnn_cell_166_biasadd_readvariableop_resource:PF
4simple_rnn_cell_166_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_166/BiasAdd/ReadVariableOp?)simple_rnn_cell_166/MatMul/ReadVariableOp?+simple_rnn_cell_166/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_166_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_166/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_166_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_166/BiasAddBiasAdd$simple_rnn_cell_166/MatMul:product:02simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_166_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_166/MatMul_1MatMulzeros:output:03simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_166/addAddV2$simple_rnn_cell_166/BiasAdd:output:0&simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_166/TanhTanhsimple_rnn_cell_166/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_166_matmul_readvariableop_resource3simple_rnn_cell_166_biasadd_readvariableop_resource4simple_rnn_cell_166_matmul_1_readvariableop_resource*
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
while_body_11018270*
condR
while_cond_11018269*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_166/BiasAdd/ReadVariableOp*^simple_rnn_cell_166/MatMul/ReadVariableOp,^simple_rnn_cell_166/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_166/BiasAdd/ReadVariableOp*simple_rnn_cell_166/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_166/MatMul/ReadVariableOp)simple_rnn_cell_166/MatMul/ReadVariableOp2Z
+simple_rnn_cell_166/MatMul_1/ReadVariableOp+simple_rnn_cell_166/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?
 simple_rnn_7_while_body_110193856
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_25
1simple_rnn_7_while_simple_rnn_7_strided_slice_1_0q
msimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resource_0:PdV
Hsimple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resource_0:d[
Isimple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0:dd
simple_rnn_7_while_identity!
simple_rnn_7_while_identity_1!
simple_rnn_7_while_identity_2!
simple_rnn_7_while_identity_3!
simple_rnn_7_while_identity_43
/simple_rnn_7_while_simple_rnn_7_strided_slice_1o
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resource:PdT
Fsimple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resource:dY
Gsimple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resource:dd??=simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOp?<simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOp?>simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp?
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_7_while_placeholderMsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
<simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
-simple_rnn_7/while/simple_rnn_cell_167/MatMulMatMul=simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
=simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
.simple_rnn_7/while/simple_rnn_cell_167/BiasAddBiasAdd7simple_rnn_7/while/simple_rnn_cell_167/MatMul:product:0Esimple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
>simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
/simple_rnn_7/while/simple_rnn_cell_167/MatMul_1MatMul simple_rnn_7_while_placeholder_2Fsimple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_7/while/simple_rnn_cell_167/addAddV27simple_rnn_7/while/simple_rnn_cell_167/BiasAdd:output:09simple_rnn_7/while/simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_7/while/simple_rnn_cell_167/TanhTanh.simple_rnn_7/while/simple_rnn_cell_167/add:z:0*
T0*'
_output_shapes
:?????????d?
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_7_while_placeholder_1simple_rnn_7_while_placeholder/simple_rnn_7/while/simple_rnn_cell_167/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_7/while/addAddV2simple_rnn_7_while_placeholder!simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_7/while/add_1AddV22simple_rnn_7_while_simple_rnn_7_while_loop_counter#simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/add_1:z:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_1Identity8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_2Identitysimple_rnn_7/while/add:z:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_3IdentityGsimple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_7/while/Identity_4Identity/simple_rnn_7/while/simple_rnn_cell_167/Tanh:y:0^simple_rnn_7/while/NoOp*
T0*'
_output_shapes
:?????????d?
simple_rnn_7/while/NoOpNoOp>^simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOp=^simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOp?^simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0"G
simple_rnn_7_while_identity_1&simple_rnn_7/while/Identity_1:output:0"G
simple_rnn_7_while_identity_2&simple_rnn_7/while/Identity_2:output:0"G
simple_rnn_7_while_identity_3&simple_rnn_7/while/Identity_3:output:0"G
simple_rnn_7_while_identity_4&simple_rnn_7/while/Identity_4:output:0"d
/simple_rnn_7_while_simple_rnn_7_strided_slice_11simple_rnn_7_while_simple_rnn_7_strided_slice_1_0"?
Fsimple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resourceHsimple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resource_0"?
Gsimple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resourceIsimple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0"?
Esimple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resourceGsimple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resource_0"?
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensormsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2~
=simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOp=simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOp2|
<simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOp<simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOp2?
>simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp>simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp: 
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
while_cond_11017698
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11017698___redundant_placeholder06
2while_while_cond_11017698___redundant_placeholder16
2while_while_cond_11017698___redundant_placeholder26
2while_while_cond_11017698___redundant_placeholder3
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
while_body_11018750
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_166_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_166_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_166_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_166_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_166/MatMul/ReadVariableOp?1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_166_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_166/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_166/BiasAddBiasAdd*while/simple_rnn_cell_166/MatMul:product:08while/simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_166/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_166/addAddV2*while/simple_rnn_cell_166/BiasAdd:output:0,while/simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_166/TanhTanh!while/simple_rnn_cell_166/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_166/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_166/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_166/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_166/MatMul/ReadVariableOp2^while/simple_rnn_cell_166/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_166_biasadd_readvariableop_resource;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_166_matmul_1_readvariableop_resource<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_166_matmul_readvariableop_resource:while_simple_rnn_cell_166_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_166/MatMul/ReadVariableOp/while/simple_rnn_cell_166/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp: 
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
?>
?
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11019640
inputs_0D
2simple_rnn_cell_166_matmul_readvariableop_resource:PA
3simple_rnn_cell_166_biasadd_readvariableop_resource:PF
4simple_rnn_cell_166_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_166/BiasAdd/ReadVariableOp?)simple_rnn_cell_166/MatMul/ReadVariableOp?+simple_rnn_cell_166/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_166_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_166/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_166_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_166/BiasAddBiasAdd$simple_rnn_cell_166/MatMul:product:02simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_166_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_166/MatMul_1MatMulzeros:output:03simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_166/addAddV2$simple_rnn_cell_166/BiasAdd:output:0&simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_166/TanhTanhsimple_rnn_cell_166/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_166_matmul_readvariableop_resource3simple_rnn_cell_166_biasadd_readvariableop_resource4simple_rnn_cell_166_matmul_1_readvariableop_resource*
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
while_body_11019574*
condR
while_cond_11019573*8
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
NoOpNoOp+^simple_rnn_cell_166/BiasAdd/ReadVariableOp*^simple_rnn_cell_166/MatMul/ReadVariableOp,^simple_rnn_cell_166/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2X
*simple_rnn_cell_166/BiasAdd/ReadVariableOp*simple_rnn_cell_166/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_166/MatMul/ReadVariableOp)simple_rnn_cell_166/MatMul/ReadVariableOp2Z
+simple_rnn_cell_166/MatMul_1/ReadVariableOp+simple_rnn_cell_166/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
/__inference_simple_rnn_7_layer_call_fn_11020002
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
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11018054o
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
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_11018963
simple_rnn_6_input'
simple_rnn_6_11018941:P#
simple_rnn_6_11018943:P'
simple_rnn_6_11018945:PP'
simple_rnn_7_11018949:Pd#
simple_rnn_7_11018951:d'
simple_rnn_7_11018953:dd"
dense_3_11018957:d
dense_3_11018959:
identity??dense_3/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?$simple_rnn_6/StatefulPartitionedCall?$simple_rnn_7/StatefulPartitionedCall?
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_6_inputsimple_rnn_6_11018941simple_rnn_6_11018943simple_rnn_6_11018945*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11018816?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_6_layer_call_and_return_conditional_losses_11018692?
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0simple_rnn_7_11018949simple_rnn_7_11018951simple_rnn_7_11018953*
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
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11018663?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_7/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
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
G__inference_dropout_7_layer_call_and_return_conditional_losses_11018539?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_3_11018957dense_3_11018959*
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
E__inference_dense_3_layer_call_and_return_conditional_losses_11018483w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_3/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_6_input
?	
?
/__inference_sequential_3_layer_call_fn_11018913
simple_rnn_6_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_11018873o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_6_input
?
?
/__inference_simple_rnn_6_layer_call_fn_11019532

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
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11018816s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_3_layer_call_fn_11020503

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
E__inference_dense_3_layer_call_and_return_conditional_losses_11018483o
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
?-
?
while_body_11019898
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_166_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_166_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_166_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_166_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_166/MatMul/ReadVariableOp?1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_166_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_166/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_166/BiasAddBiasAdd*while/simple_rnn_cell_166/MatMul:product:08while/simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_166/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_166/addAddV2*while/simple_rnn_cell_166/BiasAdd:output:0,while/simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_166/TanhTanh!while/simple_rnn_cell_166/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_166/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_166/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_166/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_166/MatMul/ReadVariableOp2^while/simple_rnn_cell_166/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_166_biasadd_readvariableop_resource;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_166_matmul_1_readvariableop_resource<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_166_matmul_readvariableop_resource:while_simple_rnn_cell_166_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_166/MatMul/ReadVariableOp/while/simple_rnn_cell_166/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp: 
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
while_cond_11020400
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11020400___redundant_placeholder06
2while_while_cond_11020400___redundant_placeholder16
2while_while_cond_11020400___redundant_placeholder26
2while_while_cond_11020400___redundant_placeholder3
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
?
e
,__inference_dropout_7_layer_call_fn_11020477

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
G__inference_dropout_7_layer_call_and_return_conditional_losses_11018539o
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
?-
?
while_body_11020185
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_167_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_167_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_167_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_167_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_167/MatMul/ReadVariableOp?1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_167_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_167/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_167/BiasAddBiasAdd*while/simple_rnn_cell_167/MatMul:product:08while/simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_167/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_167/addAddV2*while/simple_rnn_cell_167/BiasAdd:output:0,while/simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_167/TanhTanh!while/simple_rnn_cell_167/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_167/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_167/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_167/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_167/MatMul/ReadVariableOp2^while/simple_rnn_cell_167/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_167_biasadd_readvariableop_resource;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_167_matmul_1_readvariableop_resource<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_167_matmul_readvariableop_resource:while_simple_rnn_cell_167_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_167/MatMul/ReadVariableOp/while/simple_rnn_cell_167/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp: 
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
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11018098

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
??
?
$__inference__traced_restore_11020856
file_prefix1
assignvariableop_dense_3_kernel:d-
assignvariableop_1_dense_3_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: J
8assignvariableop_7_simple_rnn_6_simple_rnn_cell_6_kernel:PT
Bassignvariableop_8_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel:PPD
6assignvariableop_9_simple_rnn_6_simple_rnn_cell_6_bias:PK
9assignvariableop_10_simple_rnn_7_simple_rnn_cell_7_kernel:PdU
Cassignvariableop_11_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel:ddE
7assignvariableop_12_simple_rnn_7_simple_rnn_cell_7_bias:d#
assignvariableop_13_total: #
assignvariableop_14_count: ;
)assignvariableop_15_adam_dense_3_kernel_m:d5
'assignvariableop_16_adam_dense_3_bias_m:R
@assignvariableop_17_adam_simple_rnn_6_simple_rnn_cell_6_kernel_m:P\
Jassignvariableop_18_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_m:PPL
>assignvariableop_19_adam_simple_rnn_6_simple_rnn_cell_6_bias_m:PR
@assignvariableop_20_adam_simple_rnn_7_simple_rnn_cell_7_kernel_m:Pd\
Jassignvariableop_21_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_m:ddL
>assignvariableop_22_adam_simple_rnn_7_simple_rnn_cell_7_bias_m:d;
)assignvariableop_23_adam_dense_3_kernel_v:d5
'assignvariableop_24_adam_dense_3_bias_v:R
@assignvariableop_25_adam_simple_rnn_6_simple_rnn_cell_6_kernel_v:P\
Jassignvariableop_26_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_v:PPL
>assignvariableop_27_adam_simple_rnn_6_simple_rnn_cell_6_bias_v:PR
@assignvariableop_28_adam_simple_rnn_7_simple_rnn_cell_7_kernel_v:Pd\
Jassignvariableop_29_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_v:ddL
>assignvariableop_30_adam_simple_rnn_7_simple_rnn_cell_7_bias_v:d
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
AssignVariableOpAssignVariableOpassignvariableop_dense_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_3_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp8assignvariableop_7_simple_rnn_6_simple_rnn_cell_6_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpBassignvariableop_8_simple_rnn_6_simple_rnn_cell_6_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp6assignvariableop_9_simple_rnn_6_simple_rnn_cell_6_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp9assignvariableop_10_simple_rnn_7_simple_rnn_cell_7_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpCassignvariableop_11_simple_rnn_7_simple_rnn_cell_7_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_simple_rnn_7_simple_rnn_cell_7_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_3_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_3_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp@assignvariableop_17_adam_simple_rnn_6_simple_rnn_cell_6_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpJassignvariableop_18_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp>assignvariableop_19_adam_simple_rnn_6_simple_rnn_cell_6_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_simple_rnn_7_simple_rnn_cell_7_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpJassignvariableop_21_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp>assignvariableop_22_adam_simple_rnn_7_simple_rnn_cell_7_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_3_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_3_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp@assignvariableop_25_adam_simple_rnn_6_simple_rnn_cell_6_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpJassignvariableop_26_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp>assignvariableop_27_adam_simple_rnn_6_simple_rnn_cell_6_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp@assignvariableop_28_adam_simple_rnn_7_simple_rnn_cell_7_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpJassignvariableop_29_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp>assignvariableop_30_adam_simple_rnn_7_simple_rnn_cell_7_bias_vIdentity_30:output:0"/device:CPU:0*
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
_user_specified_namefile_prefix
?
?
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11017978

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
?
?
-sequential_3_simple_rnn_6_while_cond_11017459P
Lsequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_while_loop_counterV
Rsequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_while_maximum_iterations/
+sequential_3_simple_rnn_6_while_placeholder1
-sequential_3_simple_rnn_6_while_placeholder_11
-sequential_3_simple_rnn_6_while_placeholder_2R
Nsequential_3_simple_rnn_6_while_less_sequential_3_simple_rnn_6_strided_slice_1j
fsequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_while_cond_11017459___redundant_placeholder0j
fsequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_while_cond_11017459___redundant_placeholder1j
fsequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_while_cond_11017459___redundant_placeholder2j
fsequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_while_cond_11017459___redundant_placeholder3,
(sequential_3_simple_rnn_6_while_identity
?
$sequential_3/simple_rnn_6/while/LessLess+sequential_3_simple_rnn_6_while_placeholderNsequential_3_simple_rnn_6_while_less_sequential_3_simple_rnn_6_strided_slice_1*
T0*
_output_shapes
: 
(sequential_3/simple_rnn_6/while/IdentityIdentity(sequential_3/simple_rnn_6/while/Less:z:0*
T0
*
_output_shapes
: "]
(sequential_3_simple_rnn_6_while_identity1sequential_3/simple_rnn_6/while/Identity:output:0*(
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_11018490

inputs'
simple_rnn_6_11018337:P#
simple_rnn_6_11018339:P'
simple_rnn_6_11018341:PP'
simple_rnn_7_11018459:Pd#
simple_rnn_7_11018461:d'
simple_rnn_7_11018463:dd"
dense_3_11018484:d
dense_3_11018486:
identity??dense_3/StatefulPartitionedCall?$simple_rnn_6/StatefulPartitionedCall?$simple_rnn_7/StatefulPartitionedCall?
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_6_11018337simple_rnn_6_11018339simple_rnn_6_11018341*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11018336?
dropout_6/PartitionedCallPartitionedCall-simple_rnn_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_6_layer_call_and_return_conditional_losses_11018349?
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0simple_rnn_7_11018459simple_rnn_7_11018461simple_rnn_7_11018463*
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
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11018458?
dropout_7/PartitionedCallPartitionedCall-simple_rnn_7/StatefulPartitionedCall:output:0*
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
G__inference_dropout_7_layer_call_and_return_conditional_losses_11018471?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_3_11018484dense_3_11018486*
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
E__inference_dense_3_layer_call_and_return_conditional_losses_11018483w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_3/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_11018938
simple_rnn_6_input'
simple_rnn_6_11018916:P#
simple_rnn_6_11018918:P'
simple_rnn_6_11018920:PP'
simple_rnn_7_11018924:Pd#
simple_rnn_7_11018926:d'
simple_rnn_7_11018928:dd"
dense_3_11018932:d
dense_3_11018934:
identity??dense_3/StatefulPartitionedCall?$simple_rnn_6/StatefulPartitionedCall?$simple_rnn_7/StatefulPartitionedCall?
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_6_inputsimple_rnn_6_11018916simple_rnn_6_11018918simple_rnn_6_11018920*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11018336?
dropout_6/PartitionedCallPartitionedCall-simple_rnn_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_6_layer_call_and_return_conditional_losses_11018349?
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0simple_rnn_7_11018924simple_rnn_7_11018926simple_rnn_7_11018928*
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
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11018458?
dropout_7/PartitionedCallPartitionedCall-simple_rnn_7/StatefulPartitionedCall:output:0*
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
G__inference_dropout_7_layer_call_and_return_conditional_losses_11018471?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_3_11018932dense_3_11018934*
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
E__inference_dense_3_layer_call_and_return_conditional_losses_11018483w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_3/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_6_input
?4
?
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11018213

inputs.
simple_rnn_cell_167_11018138:Pd*
simple_rnn_cell_167_11018140:d.
simple_rnn_cell_167_11018142:dd
identity??+simple_rnn_cell_167/StatefulPartitionedCall?while;
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
+simple_rnn_cell_167/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_167_11018138simple_rnn_cell_167_11018140simple_rnn_cell_167_11018142*
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
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11018098n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_167_11018138simple_rnn_cell_167_11018140simple_rnn_cell_167_11018142*
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
while_body_11018150*
condR
while_cond_11018149*8
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
NoOpNoOp,^simple_rnn_cell_167/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2Z
+simple_rnn_cell_167/StatefulPartitionedCall+simple_rnn_cell_167/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?=
?
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11018663

inputsD
2simple_rnn_cell_167_matmul_readvariableop_resource:PdA
3simple_rnn_cell_167_biasadd_readvariableop_resource:dF
4simple_rnn_cell_167_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_167/BiasAdd/ReadVariableOp?)simple_rnn_cell_167/MatMul/ReadVariableOp?+simple_rnn_cell_167/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_167_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_167/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_167_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_167/BiasAddBiasAdd$simple_rnn_cell_167/MatMul:product:02simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_167_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_167/MatMul_1MatMulzeros:output:03simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_167/addAddV2$simple_rnn_cell_167/BiasAdd:output:0&simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_167/TanhTanhsimple_rnn_cell_167/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_167_matmul_readvariableop_resource3simple_rnn_cell_167_biasadd_readvariableop_resource4simple_rnn_cell_167_matmul_1_readvariableop_resource*
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
while_body_11018597*
condR
while_cond_11018596*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_167/BiasAdd/ReadVariableOp*^simple_rnn_cell_167/MatMul/ReadVariableOp,^simple_rnn_cell_167/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_167/BiasAdd/ReadVariableOp*simple_rnn_cell_167/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_167/MatMul/ReadVariableOp)simple_rnn_cell_167/MatMul/ReadVariableOp2Z
+simple_rnn_cell_167/MatMul_1/ReadVariableOp+simple_rnn_cell_167/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?9
?
 simple_rnn_6_while_body_110192736
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_25
1simple_rnn_6_while_simple_rnn_6_strided_slice_1_0q
msimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resource_0:PV
Hsimple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resource_0:P[
Isimple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0:PP
simple_rnn_6_while_identity!
simple_rnn_6_while_identity_1!
simple_rnn_6_while_identity_2!
simple_rnn_6_while_identity_3!
simple_rnn_6_while_identity_43
/simple_rnn_6_while_simple_rnn_6_strided_slice_1o
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resource:PT
Fsimple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resource:PY
Gsimple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resource:PP??=simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOp?<simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOp?>simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp?
Dsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
6simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_6_while_placeholderMsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
<simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
-simple_rnn_6/while/simple_rnn_cell_166/MatMulMatMul=simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
=simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
.simple_rnn_6/while/simple_rnn_cell_166/BiasAddBiasAdd7simple_rnn_6/while/simple_rnn_cell_166/MatMul:product:0Esimple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
>simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
/simple_rnn_6/while/simple_rnn_cell_166/MatMul_1MatMul simple_rnn_6_while_placeholder_2Fsimple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_6/while/simple_rnn_cell_166/addAddV27simple_rnn_6/while/simple_rnn_cell_166/BiasAdd:output:09simple_rnn_6/while/simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_6/while/simple_rnn_cell_166/TanhTanh.simple_rnn_6/while/simple_rnn_cell_166/add:z:0*
T0*'
_output_shapes
:?????????P?
7simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_6_while_placeholder_1simple_rnn_6_while_placeholder/simple_rnn_6/while/simple_rnn_cell_166/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_6/while/addAddV2simple_rnn_6_while_placeholder!simple_rnn_6/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_6/while/add_1AddV22simple_rnn_6_while_simple_rnn_6_while_loop_counter#simple_rnn_6/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_6/while/IdentityIdentitysimple_rnn_6/while/add_1:z:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_6/while/Identity_1Identity8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_6/while/Identity_2Identitysimple_rnn_6/while/add:z:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_6/while/Identity_3IdentityGsimple_rnn_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_6/while/Identity_4Identity/simple_rnn_6/while/simple_rnn_cell_166/Tanh:y:0^simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:?????????P?
simple_rnn_6/while/NoOpNoOp>^simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOp=^simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOp?^simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_6_while_identity$simple_rnn_6/while/Identity:output:0"G
simple_rnn_6_while_identity_1&simple_rnn_6/while/Identity_1:output:0"G
simple_rnn_6_while_identity_2&simple_rnn_6/while/Identity_2:output:0"G
simple_rnn_6_while_identity_3&simple_rnn_6/while/Identity_3:output:0"G
simple_rnn_6_while_identity_4&simple_rnn_6/while/Identity_4:output:0"d
/simple_rnn_6_while_simple_rnn_6_strided_slice_11simple_rnn_6_while_simple_rnn_6_strided_slice_1_0"?
Fsimple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resourceHsimple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resource_0"?
Gsimple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resourceIsimple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0"?
Esimple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resourceGsimple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resource_0"?
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensormsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2~
=simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOp=simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOp2|
<simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOp<simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOp2?
>simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp>simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp: 
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
while_cond_11019681
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11019681___redundant_placeholder06
2while_while_cond_11019681___redundant_placeholder16
2while_while_cond_11019681___redundant_placeholder26
2while_while_cond_11019681___redundant_placeholder3
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
?
?
/__inference_simple_rnn_7_layer_call_fn_11020024

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
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11018458o
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
:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?!
?
while_body_11017991
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_167_11018013_0:Pd2
$while_simple_rnn_cell_167_11018015_0:d6
$while_simple_rnn_cell_167_11018017_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_167_11018013:Pd0
"while_simple_rnn_cell_167_11018015:d4
"while_simple_rnn_cell_167_11018017:dd??1while/simple_rnn_cell_167/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
1while/simple_rnn_cell_167/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_167_11018013_0$while_simple_rnn_cell_167_11018015_0$while_simple_rnn_cell_167_11018017_0*
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
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11017978?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_167/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_167/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp2^while/simple_rnn_cell_167/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_167_11018013$while_simple_rnn_cell_167_11018013_0"J
"while_simple_rnn_cell_167_11018015$while_simple_rnn_cell_167_11018015_0"J
"while_simple_rnn_cell_167_11018017$while_simple_rnn_cell_167_11018017_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2f
1while/simple_rnn_cell_167/StatefulPartitionedCall1while/simple_rnn_cell_167/StatefulPartitionedCall: 
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

?
6__inference_simple_rnn_cell_167_layer_call_fn_11020589

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
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11017978o
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
?
e
G__inference_dropout_7_layer_call_and_return_conditional_losses_11018471

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
?
H
,__inference_dropout_7_layer_call_fn_11020472

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
G__inference_dropout_7_layer_call_and_return_conditional_losses_11018471`
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
?

?
6__inference_simple_rnn_cell_166_layer_call_fn_11020527

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
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11017686o
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
while_body_11019790
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_166_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_166_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_166_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_166_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_166/MatMul/ReadVariableOp?1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_166_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_166/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_166/BiasAddBiasAdd*while/simple_rnn_cell_166/MatMul:product:08while/simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_166/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_166/addAddV2*while/simple_rnn_cell_166/BiasAdd:output:0,while/simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_166/TanhTanh!while/simple_rnn_cell_166/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_166/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_166/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_166/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_166/MatMul/ReadVariableOp2^while/simple_rnn_cell_166/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_166_biasadd_readvariableop_resource;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_166_matmul_1_readvariableop_resource<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_166_matmul_readvariableop_resource:while_simple_rnn_cell_166_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_166/MatMul/ReadVariableOp/while/simple_rnn_cell_166/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp: 
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
?=
?
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11019856

inputsD
2simple_rnn_cell_166_matmul_readvariableop_resource:PA
3simple_rnn_cell_166_biasadd_readvariableop_resource:PF
4simple_rnn_cell_166_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_166/BiasAdd/ReadVariableOp?)simple_rnn_cell_166/MatMul/ReadVariableOp?+simple_rnn_cell_166/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_166_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_166/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_166_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_166/BiasAddBiasAdd$simple_rnn_cell_166/MatMul:product:02simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_166_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_166/MatMul_1MatMulzeros:output:03simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_166/addAddV2$simple_rnn_cell_166/BiasAdd:output:0&simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_166/TanhTanhsimple_rnn_cell_166/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_166_matmul_readvariableop_resource3simple_rnn_cell_166_biasadd_readvariableop_resource4simple_rnn_cell_166_matmul_1_readvariableop_resource*
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
while_body_11019790*
condR
while_cond_11019789*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_166/BiasAdd/ReadVariableOp*^simple_rnn_cell_166/MatMul/ReadVariableOp,^simple_rnn_cell_166/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_166/BiasAdd/ReadVariableOp*simple_rnn_cell_166/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_166/MatMul/ReadVariableOp)simple_rnn_cell_166/MatMul/ReadVariableOp2Z
+simple_rnn_cell_166/MatMul_1/ReadVariableOp+simple_rnn_cell_166/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_3_layer_call_and_return_conditional_losses_11020513

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
e
,__inference_dropout_6_layer_call_fn_11019974

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
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_6_layer_call_and_return_conditional_losses_11018692s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11017806

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
??
?	
J__inference_sequential_3_layer_call_and_return_conditional_losses_11019231

inputsQ
?simple_rnn_6_simple_rnn_cell_166_matmul_readvariableop_resource:PN
@simple_rnn_6_simple_rnn_cell_166_biasadd_readvariableop_resource:PS
Asimple_rnn_6_simple_rnn_cell_166_matmul_1_readvariableop_resource:PPQ
?simple_rnn_7_simple_rnn_cell_167_matmul_readvariableop_resource:PdN
@simple_rnn_7_simple_rnn_cell_167_biasadd_readvariableop_resource:dS
Asimple_rnn_7_simple_rnn_cell_167_matmul_1_readvariableop_resource:dd8
&dense_3_matmul_readvariableop_resource:d5
'dense_3_biasadd_readvariableop_resource:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?7simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOp?6simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOp?8simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp?simple_rnn_6/while?7simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOp?6simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOp?8simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp?simple_rnn_7/whileH
simple_rnn_6/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_6/strided_sliceStridedSlicesimple_rnn_6/Shape:output:0)simple_rnn_6/strided_slice/stack:output:0+simple_rnn_6/strided_slice/stack_1:output:0+simple_rnn_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
simple_rnn_6/zeros/packedPack#simple_rnn_6/strided_slice:output:0$simple_rnn_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_6/zerosFill"simple_rnn_6/zeros/packed:output:0!simple_rnn_6/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pp
simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_6/transpose	Transposeinputs$simple_rnn_6/transpose/perm:output:0*
T0*+
_output_shapes
:?????????^
simple_rnn_6/Shape_1Shapesimple_rnn_6/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_6/strided_slice_1StridedSlicesimple_rnn_6/Shape_1:output:0+simple_rnn_6/strided_slice_1/stack:output:0-simple_rnn_6/strided_slice_1/stack_1:output:0-simple_rnn_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_6/TensorArrayV2TensorListReserve1simple_rnn_6/TensorArrayV2/element_shape:output:0%simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
4simple_rnn_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_6/transpose:y:0Ksimple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_6/strided_slice_2StridedSlicesimple_rnn_6/transpose:y:0+simple_rnn_6/strided_slice_2/stack:output:0-simple_rnn_6/strided_slice_2/stack_1:output:0-simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
6simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp?simple_rnn_6_simple_rnn_cell_166_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
'simple_rnn_6/simple_rnn_cell_166/MatMulMatMul%simple_rnn_6/strided_slice_2:output:0>simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
7simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_6_simple_rnn_cell_166_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
(simple_rnn_6/simple_rnn_cell_166/BiasAddBiasAdd1simple_rnn_6/simple_rnn_cell_166/MatMul:product:0?simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_6_simple_rnn_cell_166_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
)simple_rnn_6/simple_rnn_cell_166/MatMul_1MatMulsimple_rnn_6/zeros:output:0@simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
$simple_rnn_6/simple_rnn_cell_166/addAddV21simple_rnn_6/simple_rnn_cell_166/BiasAdd:output:03simple_rnn_6/simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
%simple_rnn_6/simple_rnn_cell_166/TanhTanh(simple_rnn_6/simple_rnn_cell_166/add:z:0*
T0*'
_output_shapes
:?????????P{
*simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
simple_rnn_6/TensorArrayV2_1TensorListReserve3simple_rnn_6/TensorArrayV2_1/element_shape:output:0%simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_6/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_6/whileWhile(simple_rnn_6/while/loop_counter:output:0.simple_rnn_6/while/maximum_iterations:output:0simple_rnn_6/time:output:0%simple_rnn_6/TensorArrayV2_1:handle:0simple_rnn_6/zeros:output:0%simple_rnn_6/strided_slice_1:output:0Dsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_6_simple_rnn_cell_166_matmul_readvariableop_resource@simple_rnn_6_simple_rnn_cell_166_biasadd_readvariableop_resourceAsimple_rnn_6_simple_rnn_cell_166_matmul_1_readvariableop_resource*
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
 simple_rnn_6_while_body_11019053*,
cond$R"
 simple_rnn_6_while_cond_11019052*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
=simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_6/while:output:3Fsimple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0u
"simple_rnn_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_6/strided_slice_3StridedSlice8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_6/strided_slice_3/stack:output:0-simple_rnn_6/strided_slice_3/stack_1:output:0-simple_rnn_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maskr
simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_6/transpose_1	Transpose8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pr
dropout_6/IdentityIdentitysimple_rnn_6/transpose_1:y:0*
T0*+
_output_shapes
:?????????P]
simple_rnn_7/ShapeShapedropout_6/Identity:output:0*
T0*
_output_shapes
:j
 simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_sliceStridedSlicesimple_rnn_7/Shape:output:0)simple_rnn_7/strided_slice/stack:output:0+simple_rnn_7/strided_slice/stack_1:output:0+simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
simple_rnn_7/zeros/packedPack#simple_rnn_7/strided_slice:output:0$simple_rnn_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_7/zerosFill"simple_rnn_7/zeros/packed:output:0!simple_rnn_7/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dp
simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_7/transpose	Transposedropout_6/Identity:output:0$simple_rnn_7/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P^
simple_rnn_7/Shape_1Shapesimple_rnn_7/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_slice_1StridedSlicesimple_rnn_7/Shape_1:output:0+simple_rnn_7/strided_slice_1/stack:output:0-simple_rnn_7/strided_slice_1/stack_1:output:0-simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_7/TensorArrayV2TensorListReserve1simple_rnn_7/TensorArrayV2/element_shape:output:0%simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_7/transpose:y:0Ksimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_slice_2StridedSlicesimple_rnn_7/transpose:y:0+simple_rnn_7/strided_slice_2/stack:output:0-simple_rnn_7/strided_slice_2/stack_1:output:0-simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
6simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_167_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
'simple_rnn_7/simple_rnn_cell_167/MatMulMatMul%simple_rnn_7/strided_slice_2:output:0>simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
7simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_7_simple_rnn_cell_167_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
(simple_rnn_7/simple_rnn_cell_167/BiasAddBiasAdd1simple_rnn_7/simple_rnn_cell_167/MatMul:product:0?simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_7_simple_rnn_cell_167_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
)simple_rnn_7/simple_rnn_cell_167/MatMul_1MatMulsimple_rnn_7/zeros:output:0@simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
$simple_rnn_7/simple_rnn_cell_167/addAddV21simple_rnn_7/simple_rnn_cell_167/BiasAdd:output:03simple_rnn_7/simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
%simple_rnn_7/simple_rnn_cell_167/TanhTanh(simple_rnn_7/simple_rnn_cell_167/add:z:0*
T0*'
_output_shapes
:?????????d{
*simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
simple_rnn_7/TensorArrayV2_1TensorListReserve3simple_rnn_7/TensorArrayV2_1/element_shape:output:0%simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_7/whileWhile(simple_rnn_7/while/loop_counter:output:0.simple_rnn_7/while/maximum_iterations:output:0simple_rnn_7/time:output:0%simple_rnn_7/TensorArrayV2_1:handle:0simple_rnn_7/zeros:output:0%simple_rnn_7/strided_slice_1:output:0Dsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_7_simple_rnn_cell_167_matmul_readvariableop_resource@simple_rnn_7_simple_rnn_cell_167_biasadd_readvariableop_resourceAsimple_rnn_7_simple_rnn_cell_167_matmul_1_readvariableop_resource*
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
 simple_rnn_7_while_body_11019158*,
cond$R"
 simple_rnn_7_while_cond_11019157*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_7/while:output:3Fsimple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0u
"simple_rnn_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_slice_3StridedSlice8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_7/strided_slice_3/stack:output:0-simple_rnn_7/strided_slice_3/stack_1:output:0-simple_rnn_7/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maskr
simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_7/transpose_1	Transpose8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_7/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dw
dropout_7/IdentityIdentity%simple_rnn_7/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_3/MatMulMatMuldropout_7/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp8^simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOp7^simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOp9^simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp^simple_rnn_6/while8^simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOp7^simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOp9^simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp^simple_rnn_7/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2r
7simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOp7simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOp2p
6simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOp6simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOp2t
8simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp8simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp2(
simple_rnn_6/whilesimple_rnn_6/while2r
7simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOp7simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOp2p
6simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOp6simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOp2t
8simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp8simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp2(
simple_rnn_7/whilesimple_rnn_7/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_11020184
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11020184___redundant_placeholder06
2while_while_cond_11020184___redundant_placeholder16
2while_while_cond_11020184___redundant_placeholder26
2while_while_cond_11020184___redundant_placeholder3
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
while_cond_11020076
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11020076___redundant_placeholder06
2while_while_cond_11020076___redundant_placeholder16
2while_while_cond_11020076___redundant_placeholder26
2while_while_cond_11020076___redundant_placeholder3
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
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11020251
inputs_0D
2simple_rnn_cell_167_matmul_readvariableop_resource:PdA
3simple_rnn_cell_167_biasadd_readvariableop_resource:dF
4simple_rnn_cell_167_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_167/BiasAdd/ReadVariableOp?)simple_rnn_cell_167/MatMul/ReadVariableOp?+simple_rnn_cell_167/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_167_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_167/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_167_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_167/BiasAddBiasAdd$simple_rnn_cell_167/MatMul:product:02simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_167_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_167/MatMul_1MatMulzeros:output:03simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_167/addAddV2$simple_rnn_cell_167/BiasAdd:output:0&simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_167/TanhTanhsimple_rnn_cell_167/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_167_matmul_readvariableop_resource3simple_rnn_cell_167_biasadd_readvariableop_resource4simple_rnn_cell_167_matmul_1_readvariableop_resource*
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
while_body_11020185*
condR
while_cond_11020184*8
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
NoOpNoOp+^simple_rnn_cell_167/BiasAdd/ReadVariableOp*^simple_rnn_cell_167/MatMul/ReadVariableOp,^simple_rnn_cell_167/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2X
*simple_rnn_cell_167/BiasAdd/ReadVariableOp*simple_rnn_cell_167/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_167/MatMul/ReadVariableOp)simple_rnn_cell_167/MatMul/ReadVariableOp2Z
+simple_rnn_cell_167/MatMul_1/ReadVariableOp+simple_rnn_cell_167/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?E
?
-sequential_3_simple_rnn_7_while_body_11017565P
Lsequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_while_loop_counterV
Rsequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_while_maximum_iterations/
+sequential_3_simple_rnn_7_while_placeholder1
-sequential_3_simple_rnn_7_while_placeholder_11
-sequential_3_simple_rnn_7_while_placeholder_2O
Ksequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_strided_slice_1_0?
?sequential_3_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0f
Tsequential_3_simple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resource_0:Pdc
Usequential_3_simple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resource_0:dh
Vsequential_3_simple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0:dd,
(sequential_3_simple_rnn_7_while_identity.
*sequential_3_simple_rnn_7_while_identity_1.
*sequential_3_simple_rnn_7_while_identity_2.
*sequential_3_simple_rnn_7_while_identity_3.
*sequential_3_simple_rnn_7_while_identity_4M
Isequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_strided_slice_1?
?sequential_3_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_7_tensorarrayunstack_tensorlistfromtensord
Rsequential_3_simple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resource:Pda
Ssequential_3_simple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resource:df
Tsequential_3_simple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resource:dd??Jsequential_3/simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOp?Isequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOp?Ksequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp?
Qsequential_3/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
Csequential_3/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_3_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0+sequential_3_simple_rnn_7_while_placeholderZsequential_3/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
Isequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOpTsequential_3_simple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
:sequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMulMatMulJsequential_3/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0Qsequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Jsequential_3/simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOpUsequential_3_simple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
;sequential_3/simple_rnn_7/while/simple_rnn_cell_167/BiasAddBiasAddDsequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul:product:0Rsequential_3/simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Ksequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOpVsequential_3_simple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
<sequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul_1MatMul-sequential_3_simple_rnn_7_while_placeholder_2Ssequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
7sequential_3/simple_rnn_7/while/simple_rnn_cell_167/addAddV2Dsequential_3/simple_rnn_7/while/simple_rnn_cell_167/BiasAdd:output:0Fsequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
8sequential_3/simple_rnn_7/while/simple_rnn_cell_167/TanhTanh;sequential_3/simple_rnn_7/while/simple_rnn_cell_167/add:z:0*
T0*'
_output_shapes
:?????????d?
Dsequential_3/simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_3_simple_rnn_7_while_placeholder_1+sequential_3_simple_rnn_7_while_placeholder<sequential_3/simple_rnn_7/while/simple_rnn_cell_167/Tanh:y:0*
_output_shapes
: *
element_dtype0:???g
%sequential_3/simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
#sequential_3/simple_rnn_7/while/addAddV2+sequential_3_simple_rnn_7_while_placeholder.sequential_3/simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: i
'sequential_3/simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
%sequential_3/simple_rnn_7/while/add_1AddV2Lsequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_while_loop_counter0sequential_3/simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: ?
(sequential_3/simple_rnn_7/while/IdentityIdentity)sequential_3/simple_rnn_7/while/add_1:z:0%^sequential_3/simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
*sequential_3/simple_rnn_7/while/Identity_1IdentityRsequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_while_maximum_iterations%^sequential_3/simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
*sequential_3/simple_rnn_7/while/Identity_2Identity'sequential_3/simple_rnn_7/while/add:z:0%^sequential_3/simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
*sequential_3/simple_rnn_7/while/Identity_3IdentityTsequential_3/simple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^sequential_3/simple_rnn_7/while/NoOp*
T0*
_output_shapes
: :????
*sequential_3/simple_rnn_7/while/Identity_4Identity<sequential_3/simple_rnn_7/while/simple_rnn_cell_167/Tanh:y:0%^sequential_3/simple_rnn_7/while/NoOp*
T0*'
_output_shapes
:?????????d?
$sequential_3/simple_rnn_7/while/NoOpNoOpK^sequential_3/simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOpJ^sequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOpL^sequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_3_simple_rnn_7_while_identity1sequential_3/simple_rnn_7/while/Identity:output:0"a
*sequential_3_simple_rnn_7_while_identity_13sequential_3/simple_rnn_7/while/Identity_1:output:0"a
*sequential_3_simple_rnn_7_while_identity_23sequential_3/simple_rnn_7/while/Identity_2:output:0"a
*sequential_3_simple_rnn_7_while_identity_33sequential_3/simple_rnn_7/while/Identity_3:output:0"a
*sequential_3_simple_rnn_7_while_identity_43sequential_3/simple_rnn_7/while/Identity_4:output:0"?
Isequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_strided_slice_1Ksequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_strided_slice_1_0"?
Ssequential_3_simple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resourceUsequential_3_simple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resource_0"?
Tsequential_3_simple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resourceVsequential_3_simple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0"?
Rsequential_3_simple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resourceTsequential_3_simple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resource_0"?
?sequential_3_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor?sequential_3_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2?
Jsequential_3/simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOpJsequential_3/simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOp2?
Isequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOpIsequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOp2?
Ksequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOpKsequential_3/simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp: 
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
ҵ
?

#__inference__wrapped_model_11017638
simple_rnn_6_input^
Lsequential_3_simple_rnn_6_simple_rnn_cell_166_matmul_readvariableop_resource:P[
Msequential_3_simple_rnn_6_simple_rnn_cell_166_biasadd_readvariableop_resource:P`
Nsequential_3_simple_rnn_6_simple_rnn_cell_166_matmul_1_readvariableop_resource:PP^
Lsequential_3_simple_rnn_7_simple_rnn_cell_167_matmul_readvariableop_resource:Pd[
Msequential_3_simple_rnn_7_simple_rnn_cell_167_biasadd_readvariableop_resource:d`
Nsequential_3_simple_rnn_7_simple_rnn_cell_167_matmul_1_readvariableop_resource:ddE
3sequential_3_dense_3_matmul_readvariableop_resource:dB
4sequential_3_dense_3_biasadd_readvariableop_resource:
identity??+sequential_3/dense_3/BiasAdd/ReadVariableOp?*sequential_3/dense_3/MatMul/ReadVariableOp?Dsequential_3/simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOp?Csequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOp?Esequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp?sequential_3/simple_rnn_6/while?Dsequential_3/simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOp?Csequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOp?Esequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp?sequential_3/simple_rnn_7/whilea
sequential_3/simple_rnn_6/ShapeShapesimple_rnn_6_input*
T0*
_output_shapes
:w
-sequential_3/simple_rnn_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential_3/simple_rnn_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential_3/simple_rnn_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'sequential_3/simple_rnn_6/strided_sliceStridedSlice(sequential_3/simple_rnn_6/Shape:output:06sequential_3/simple_rnn_6/strided_slice/stack:output:08sequential_3/simple_rnn_6/strided_slice/stack_1:output:08sequential_3/simple_rnn_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_3/simple_rnn_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
&sequential_3/simple_rnn_6/zeros/packedPack0sequential_3/simple_rnn_6/strided_slice:output:01sequential_3/simple_rnn_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%sequential_3/simple_rnn_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_3/simple_rnn_6/zerosFill/sequential_3/simple_rnn_6/zeros/packed:output:0.sequential_3/simple_rnn_6/zeros/Const:output:0*
T0*'
_output_shapes
:?????????P}
(sequential_3/simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
#sequential_3/simple_rnn_6/transpose	Transposesimple_rnn_6_input1sequential_3/simple_rnn_6/transpose/perm:output:0*
T0*+
_output_shapes
:?????????x
!sequential_3/simple_rnn_6/Shape_1Shape'sequential_3/simple_rnn_6/transpose:y:0*
T0*
_output_shapes
:y
/sequential_3/simple_rnn_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_3/simple_rnn_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_3/simple_rnn_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_3/simple_rnn_6/strided_slice_1StridedSlice*sequential_3/simple_rnn_6/Shape_1:output:08sequential_3/simple_rnn_6/strided_slice_1/stack:output:0:sequential_3/simple_rnn_6/strided_slice_1/stack_1:output:0:sequential_3/simple_rnn_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
5sequential_3/simple_rnn_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'sequential_3/simple_rnn_6/TensorArrayV2TensorListReserve>sequential_3/simple_rnn_6/TensorArrayV2/element_shape:output:02sequential_3/simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Osequential_3/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Asequential_3/simple_rnn_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_3/simple_rnn_6/transpose:y:0Xsequential_3/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???y
/sequential_3/simple_rnn_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_3/simple_rnn_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_3/simple_rnn_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_3/simple_rnn_6/strided_slice_2StridedSlice'sequential_3/simple_rnn_6/transpose:y:08sequential_3/simple_rnn_6/strided_slice_2/stack:output:0:sequential_3/simple_rnn_6/strided_slice_2/stack_1:output:0:sequential_3/simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
Csequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOpLsequential_3_simple_rnn_6_simple_rnn_cell_166_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
4sequential_3/simple_rnn_6/simple_rnn_cell_166/MatMulMatMul2sequential_3/simple_rnn_6/strided_slice_2:output:0Ksequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Dsequential_3/simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOpMsequential_3_simple_rnn_6_simple_rnn_cell_166_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
5sequential_3/simple_rnn_6/simple_rnn_cell_166/BiasAddBiasAdd>sequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul:product:0Lsequential_3/simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Esequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOpNsequential_3_simple_rnn_6_simple_rnn_cell_166_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
6sequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul_1MatMul(sequential_3/simple_rnn_6/zeros:output:0Msequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1sequential_3/simple_rnn_6/simple_rnn_cell_166/addAddV2>sequential_3/simple_rnn_6/simple_rnn_cell_166/BiasAdd:output:0@sequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
2sequential_3/simple_rnn_6/simple_rnn_cell_166/TanhTanh5sequential_3/simple_rnn_6/simple_rnn_cell_166/add:z:0*
T0*'
_output_shapes
:?????????P?
7sequential_3/simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)sequential_3/simple_rnn_6/TensorArrayV2_1TensorListReserve@sequential_3/simple_rnn_6/TensorArrayV2_1/element_shape:output:02sequential_3/simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???`
sequential_3/simple_rnn_6/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2sequential_3/simple_rnn_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????n
,sequential_3/simple_rnn_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_3/simple_rnn_6/whileWhile5sequential_3/simple_rnn_6/while/loop_counter:output:0;sequential_3/simple_rnn_6/while/maximum_iterations:output:0'sequential_3/simple_rnn_6/time:output:02sequential_3/simple_rnn_6/TensorArrayV2_1:handle:0(sequential_3/simple_rnn_6/zeros:output:02sequential_3/simple_rnn_6/strided_slice_1:output:0Qsequential_3/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0Lsequential_3_simple_rnn_6_simple_rnn_cell_166_matmul_readvariableop_resourceMsequential_3_simple_rnn_6_simple_rnn_cell_166_biasadd_readvariableop_resourceNsequential_3_simple_rnn_6_simple_rnn_cell_166_matmul_1_readvariableop_resource*
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
-sequential_3_simple_rnn_6_while_body_11017460*9
cond1R/
-sequential_3_simple_rnn_6_while_cond_11017459*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
Jsequential_3/simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
<sequential_3/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_3/simple_rnn_6/while:output:3Ssequential_3/simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0?
/sequential_3/simple_rnn_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1sequential_3/simple_rnn_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1sequential_3/simple_rnn_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_3/simple_rnn_6/strided_slice_3StridedSliceEsequential_3/simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:08sequential_3/simple_rnn_6/strided_slice_3/stack:output:0:sequential_3/simple_rnn_6/strided_slice_3/stack_1:output:0:sequential_3/simple_rnn_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask
*sequential_3/simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
%sequential_3/simple_rnn_6/transpose_1	TransposeEsequential_3/simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:03sequential_3/simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P?
sequential_3/dropout_6/IdentityIdentity)sequential_3/simple_rnn_6/transpose_1:y:0*
T0*+
_output_shapes
:?????????Pw
sequential_3/simple_rnn_7/ShapeShape(sequential_3/dropout_6/Identity:output:0*
T0*
_output_shapes
:w
-sequential_3/simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential_3/simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential_3/simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'sequential_3/simple_rnn_7/strided_sliceStridedSlice(sequential_3/simple_rnn_7/Shape:output:06sequential_3/simple_rnn_7/strided_slice/stack:output:08sequential_3/simple_rnn_7/strided_slice/stack_1:output:08sequential_3/simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_3/simple_rnn_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
&sequential_3/simple_rnn_7/zeros/packedPack0sequential_3/simple_rnn_7/strided_slice:output:01sequential_3/simple_rnn_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%sequential_3/simple_rnn_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_3/simple_rnn_7/zerosFill/sequential_3/simple_rnn_7/zeros/packed:output:0.sequential_3/simple_rnn_7/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d}
(sequential_3/simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
#sequential_3/simple_rnn_7/transpose	Transpose(sequential_3/dropout_6/Identity:output:01sequential_3/simple_rnn_7/transpose/perm:output:0*
T0*+
_output_shapes
:?????????Px
!sequential_3/simple_rnn_7/Shape_1Shape'sequential_3/simple_rnn_7/transpose:y:0*
T0*
_output_shapes
:y
/sequential_3/simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_3/simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_3/simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_3/simple_rnn_7/strided_slice_1StridedSlice*sequential_3/simple_rnn_7/Shape_1:output:08sequential_3/simple_rnn_7/strided_slice_1/stack:output:0:sequential_3/simple_rnn_7/strided_slice_1/stack_1:output:0:sequential_3/simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
5sequential_3/simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'sequential_3/simple_rnn_7/TensorArrayV2TensorListReserve>sequential_3/simple_rnn_7/TensorArrayV2/element_shape:output:02sequential_3/simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Osequential_3/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
Asequential_3/simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_3/simple_rnn_7/transpose:y:0Xsequential_3/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???y
/sequential_3/simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_3/simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_3/simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_3/simple_rnn_7/strided_slice_2StridedSlice'sequential_3/simple_rnn_7/transpose:y:08sequential_3/simple_rnn_7/strided_slice_2/stack:output:0:sequential_3/simple_rnn_7/strided_slice_2/stack_1:output:0:sequential_3/simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
Csequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOpLsequential_3_simple_rnn_7_simple_rnn_cell_167_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
4sequential_3/simple_rnn_7/simple_rnn_cell_167/MatMulMatMul2sequential_3/simple_rnn_7/strided_slice_2:output:0Ksequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Dsequential_3/simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOpMsequential_3_simple_rnn_7_simple_rnn_cell_167_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
5sequential_3/simple_rnn_7/simple_rnn_cell_167/BiasAddBiasAdd>sequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul:product:0Lsequential_3/simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Esequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOpNsequential_3_simple_rnn_7_simple_rnn_cell_167_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
6sequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul_1MatMul(sequential_3/simple_rnn_7/zeros:output:0Msequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1sequential_3/simple_rnn_7/simple_rnn_cell_167/addAddV2>sequential_3/simple_rnn_7/simple_rnn_cell_167/BiasAdd:output:0@sequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
2sequential_3/simple_rnn_7/simple_rnn_cell_167/TanhTanh5sequential_3/simple_rnn_7/simple_rnn_cell_167/add:z:0*
T0*'
_output_shapes
:?????????d?
7sequential_3/simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
)sequential_3/simple_rnn_7/TensorArrayV2_1TensorListReserve@sequential_3/simple_rnn_7/TensorArrayV2_1/element_shape:output:02sequential_3/simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???`
sequential_3/simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2sequential_3/simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????n
,sequential_3/simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_3/simple_rnn_7/whileWhile5sequential_3/simple_rnn_7/while/loop_counter:output:0;sequential_3/simple_rnn_7/while/maximum_iterations:output:0'sequential_3/simple_rnn_7/time:output:02sequential_3/simple_rnn_7/TensorArrayV2_1:handle:0(sequential_3/simple_rnn_7/zeros:output:02sequential_3/simple_rnn_7/strided_slice_1:output:0Qsequential_3/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0Lsequential_3_simple_rnn_7_simple_rnn_cell_167_matmul_readvariableop_resourceMsequential_3_simple_rnn_7_simple_rnn_cell_167_biasadd_readvariableop_resourceNsequential_3_simple_rnn_7_simple_rnn_cell_167_matmul_1_readvariableop_resource*
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
-sequential_3_simple_rnn_7_while_body_11017565*9
cond1R/
-sequential_3_simple_rnn_7_while_cond_11017564*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
Jsequential_3/simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
<sequential_3/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_3/simple_rnn_7/while:output:3Ssequential_3/simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0?
/sequential_3/simple_rnn_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1sequential_3/simple_rnn_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1sequential_3/simple_rnn_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_3/simple_rnn_7/strided_slice_3StridedSliceEsequential_3/simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:08sequential_3/simple_rnn_7/strided_slice_3/stack:output:0:sequential_3/simple_rnn_7/strided_slice_3/stack_1:output:0:sequential_3/simple_rnn_7/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask
*sequential_3/simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
%sequential_3/simple_rnn_7/transpose_1	TransposeEsequential_3/simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:03sequential_3/simple_rnn_7/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d?
sequential_3/dropout_7/IdentityIdentity2sequential_3/simple_rnn_7/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
*sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
sequential_3/dense_3/MatMulMatMul(sequential_3/dropout_7/Identity:output:02sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/dense_3/BiasAddBiasAdd%sequential_3/dense_3/MatMul:product:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_3/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^sequential_3/dense_3/BiasAdd/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOpE^sequential_3/simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOpD^sequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOpF^sequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp ^sequential_3/simple_rnn_6/whileE^sequential_3/simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOpD^sequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOpF^sequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp ^sequential_3/simple_rnn_7/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2X
*sequential_3/dense_3/MatMul/ReadVariableOp*sequential_3/dense_3/MatMul/ReadVariableOp2?
Dsequential_3/simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOpDsequential_3/simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOp2?
Csequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOpCsequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOp2?
Esequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOpEsequential_3/simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp2B
sequential_3/simple_rnn_6/whilesequential_3/simple_rnn_6/while2?
Dsequential_3/simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOpDsequential_3/simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOp2?
Csequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOpCsequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOp2?
Esequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOpEsequential_3/simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp2B
sequential_3/simple_rnn_7/whilesequential_3/simple_rnn_7/while:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_6_input
?
?
while_cond_11019573
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11019573___redundant_placeholder06
2while_while_cond_11019573___redundant_placeholder16
2while_while_cond_11019573___redundant_placeholder26
2while_while_cond_11019573___redundant_placeholder3
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
while_cond_11018596
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11018596___redundant_placeholder06
2while_while_cond_11018596___redundant_placeholder16
2while_while_cond_11018596___redundant_placeholder26
2while_while_cond_11018596___redundant_placeholder3
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
?E
?
-sequential_3_simple_rnn_6_while_body_11017460P
Lsequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_while_loop_counterV
Rsequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_while_maximum_iterations/
+sequential_3_simple_rnn_6_while_placeholder1
-sequential_3_simple_rnn_6_while_placeholder_11
-sequential_3_simple_rnn_6_while_placeholder_2O
Ksequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_strided_slice_1_0?
?sequential_3_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0f
Tsequential_3_simple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resource_0:Pc
Usequential_3_simple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resource_0:Ph
Vsequential_3_simple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0:PP,
(sequential_3_simple_rnn_6_while_identity.
*sequential_3_simple_rnn_6_while_identity_1.
*sequential_3_simple_rnn_6_while_identity_2.
*sequential_3_simple_rnn_6_while_identity_3.
*sequential_3_simple_rnn_6_while_identity_4M
Isequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_strided_slice_1?
?sequential_3_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_6_tensorarrayunstack_tensorlistfromtensord
Rsequential_3_simple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resource:Pa
Ssequential_3_simple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resource:Pf
Tsequential_3_simple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resource:PP??Jsequential_3/simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOp?Isequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOp?Ksequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp?
Qsequential_3/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Csequential_3/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_3_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0+sequential_3_simple_rnn_6_while_placeholderZsequential_3/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
Isequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOpTsequential_3_simple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
:sequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMulMatMulJsequential_3/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Qsequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Jsequential_3/simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOpUsequential_3_simple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
;sequential_3/simple_rnn_6/while/simple_rnn_cell_166/BiasAddBiasAddDsequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul:product:0Rsequential_3/simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Ksequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOpVsequential_3_simple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
<sequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul_1MatMul-sequential_3_simple_rnn_6_while_placeholder_2Ssequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
7sequential_3/simple_rnn_6/while/simple_rnn_cell_166/addAddV2Dsequential_3/simple_rnn_6/while/simple_rnn_cell_166/BiasAdd:output:0Fsequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
8sequential_3/simple_rnn_6/while/simple_rnn_cell_166/TanhTanh;sequential_3/simple_rnn_6/while/simple_rnn_cell_166/add:z:0*
T0*'
_output_shapes
:?????????P?
Dsequential_3/simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_3_simple_rnn_6_while_placeholder_1+sequential_3_simple_rnn_6_while_placeholder<sequential_3/simple_rnn_6/while/simple_rnn_cell_166/Tanh:y:0*
_output_shapes
: *
element_dtype0:???g
%sequential_3/simple_rnn_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
#sequential_3/simple_rnn_6/while/addAddV2+sequential_3_simple_rnn_6_while_placeholder.sequential_3/simple_rnn_6/while/add/y:output:0*
T0*
_output_shapes
: i
'sequential_3/simple_rnn_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
%sequential_3/simple_rnn_6/while/add_1AddV2Lsequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_while_loop_counter0sequential_3/simple_rnn_6/while/add_1/y:output:0*
T0*
_output_shapes
: ?
(sequential_3/simple_rnn_6/while/IdentityIdentity)sequential_3/simple_rnn_6/while/add_1:z:0%^sequential_3/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ?
*sequential_3/simple_rnn_6/while/Identity_1IdentityRsequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_while_maximum_iterations%^sequential_3/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ?
*sequential_3/simple_rnn_6/while/Identity_2Identity'sequential_3/simple_rnn_6/while/add:z:0%^sequential_3/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ?
*sequential_3/simple_rnn_6/while/Identity_3IdentityTsequential_3/simple_rnn_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^sequential_3/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: :????
*sequential_3/simple_rnn_6/while/Identity_4Identity<sequential_3/simple_rnn_6/while/simple_rnn_cell_166/Tanh:y:0%^sequential_3/simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:?????????P?
$sequential_3/simple_rnn_6/while/NoOpNoOpK^sequential_3/simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOpJ^sequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOpL^sequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_3_simple_rnn_6_while_identity1sequential_3/simple_rnn_6/while/Identity:output:0"a
*sequential_3_simple_rnn_6_while_identity_13sequential_3/simple_rnn_6/while/Identity_1:output:0"a
*sequential_3_simple_rnn_6_while_identity_23sequential_3/simple_rnn_6/while/Identity_2:output:0"a
*sequential_3_simple_rnn_6_while_identity_33sequential_3/simple_rnn_6/while/Identity_3:output:0"a
*sequential_3_simple_rnn_6_while_identity_43sequential_3/simple_rnn_6/while/Identity_4:output:0"?
Isequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_strided_slice_1Ksequential_3_simple_rnn_6_while_sequential_3_simple_rnn_6_strided_slice_1_0"?
Ssequential_3_simple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resourceUsequential_3_simple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resource_0"?
Tsequential_3_simple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resourceVsequential_3_simple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0"?
Rsequential_3_simple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resourceTsequential_3_simple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resource_0"?
?sequential_3_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor?sequential_3_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_3_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2?
Jsequential_3/simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOpJsequential_3/simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOp2?
Isequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOpIsequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOp2?
Ksequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOpKsequential_3/simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp: 
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
while_cond_11019897
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11019897___redundant_placeholder06
2while_while_cond_11019897___redundant_placeholder16
2while_while_cond_11019897___redundant_placeholder26
2while_while_cond_11019897___redundant_placeholder3
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
while_body_11020401
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_167_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_167_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_167_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_167_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_167/MatMul/ReadVariableOp?1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_167_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_167/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_167/BiasAddBiasAdd*while/simple_rnn_cell_167/MatMul:product:08while/simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_167/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_167/addAddV2*while/simple_rnn_cell_167/BiasAdd:output:0,while/simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_167/TanhTanh!while/simple_rnn_cell_167/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_167/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_167/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_167/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_167/MatMul/ReadVariableOp2^while/simple_rnn_cell_167/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_167_biasadd_readvariableop_resource;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_167_matmul_1_readvariableop_resource<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_167_matmul_readvariableop_resource:while_simple_rnn_cell_167_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_167/MatMul/ReadVariableOp/while/simple_rnn_cell_167/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp: 
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

f
G__inference_dropout_6_layer_call_and_return_conditional_losses_11019991

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
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
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
:?????????Ps
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Pm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?=
?
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11018816

inputsD
2simple_rnn_cell_166_matmul_readvariableop_resource:PA
3simple_rnn_cell_166_biasadd_readvariableop_resource:PF
4simple_rnn_cell_166_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_166/BiasAdd/ReadVariableOp?)simple_rnn_cell_166/MatMul/ReadVariableOp?+simple_rnn_cell_166/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_166_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_166/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_166_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_166/BiasAddBiasAdd$simple_rnn_cell_166/MatMul:product:02simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_166_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_166/MatMul_1MatMulzeros:output:03simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_166/addAddV2$simple_rnn_cell_166/BiasAdd:output:0&simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_166/TanhTanhsimple_rnn_cell_166/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_166_matmul_readvariableop_resource3simple_rnn_cell_166_biasadd_readvariableop_resource4simple_rnn_cell_166_matmul_1_readvariableop_resource*
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
while_body_11018750*
condR
while_cond_11018749*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_166/BiasAdd/ReadVariableOp*^simple_rnn_cell_166/MatMul/ReadVariableOp,^simple_rnn_cell_166/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_166/BiasAdd/ReadVariableOp*simple_rnn_cell_166/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_166/MatMul/ReadVariableOp)simple_rnn_cell_166/MatMul/ReadVariableOp2Z
+simple_rnn_cell_166/MatMul_1/ReadVariableOp+simple_rnn_cell_166/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_6_layer_call_and_return_conditional_losses_11018349

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
/__inference_simple_rnn_7_layer_call_fn_11020035

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
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11018663o
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
:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?	
?
/__inference_sequential_3_layer_call_fn_11018509
simple_rnn_6_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_11018490o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_6_input
?
?
while_cond_11018391
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11018391___redundant_placeholder06
2while_while_cond_11018391___redundant_placeholder16
2while_while_cond_11018391___redundant_placeholder26
2while_while_cond_11018391___redundant_placeholder3
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
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11019748
inputs_0D
2simple_rnn_cell_166_matmul_readvariableop_resource:PA
3simple_rnn_cell_166_biasadd_readvariableop_resource:PF
4simple_rnn_cell_166_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_166/BiasAdd/ReadVariableOp?)simple_rnn_cell_166/MatMul/ReadVariableOp?+simple_rnn_cell_166/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_166_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_166/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_166_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_166/BiasAddBiasAdd$simple_rnn_cell_166/MatMul:product:02simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_166_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_166/MatMul_1MatMulzeros:output:03simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_166/addAddV2$simple_rnn_cell_166/BiasAdd:output:0&simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_166/TanhTanhsimple_rnn_cell_166/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_166_matmul_readvariableop_resource3simple_rnn_cell_166_biasadd_readvariableop_resource4simple_rnn_cell_166_matmul_1_readvariableop_resource*
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
while_body_11019682*
condR
while_cond_11019681*8
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
NoOpNoOp+^simple_rnn_cell_166/BiasAdd/ReadVariableOp*^simple_rnn_cell_166/MatMul/ReadVariableOp,^simple_rnn_cell_166/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2X
*simple_rnn_cell_166/BiasAdd/ReadVariableOp*simple_rnn_cell_166/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_166/MatMul/ReadVariableOp)simple_rnn_cell_166/MatMul/ReadVariableOp2Z
+simple_rnn_cell_166/MatMul_1/ReadVariableOp+simple_rnn_cell_166/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?

?
 simple_rnn_7_while_cond_110193846
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_28
4simple_rnn_7_while_less_simple_rnn_7_strided_slice_1P
Lsimple_rnn_7_while_simple_rnn_7_while_cond_11019384___redundant_placeholder0P
Lsimple_rnn_7_while_simple_rnn_7_while_cond_11019384___redundant_placeholder1P
Lsimple_rnn_7_while_simple_rnn_7_while_cond_11019384___redundant_placeholder2P
Lsimple_rnn_7_while_simple_rnn_7_while_cond_11019384___redundant_placeholder3
simple_rnn_7_while_identity
?
simple_rnn_7/while/LessLesssimple_rnn_7_while_placeholder4simple_rnn_7_while_less_simple_rnn_7_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0*(
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
?4
?
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11018054

inputs.
simple_rnn_cell_167_11017979:Pd*
simple_rnn_cell_167_11017981:d.
simple_rnn_cell_167_11017983:dd
identity??+simple_rnn_cell_167/StatefulPartitionedCall?while;
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
+simple_rnn_cell_167/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_167_11017979simple_rnn_cell_167_11017981simple_rnn_cell_167_11017983*
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
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11017978n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_167_11017979simple_rnn_cell_167_11017981simple_rnn_cell_167_11017983*
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
while_body_11017991*
condR
while_cond_11017990*8
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
NoOpNoOp,^simple_rnn_cell_167/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2Z
+simple_rnn_cell_167/StatefulPartitionedCall+simple_rnn_cell_167/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?H
?
!__inference__traced_save_11020753
file_prefix-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_simple_rnn_6_simple_rnn_cell_6_kernel_read_readvariableopN
Jsavev2_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_6_simple_rnn_cell_6_bias_read_readvariableopD
@savev2_simple_rnn_7_simple_rnn_cell_7_kernel_read_readvariableopN
Jsavev2_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_7_simple_rnn_cell_7_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_6_simple_rnn_cell_6_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_6_simple_rnn_cell_6_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_7_simple_rnn_cell_7_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_7_simple_rnn_cell_7_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_6_simple_rnn_cell_6_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_6_simple_rnn_cell_6_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_7_simple_rnn_cell_7_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_7_simple_rnn_cell_7_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_simple_rnn_6_simple_rnn_cell_6_kernel_read_readvariableopJsavev2_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_read_readvariableop>savev2_simple_rnn_6_simple_rnn_cell_6_bias_read_readvariableop@savev2_simple_rnn_7_simple_rnn_cell_7_kernel_read_readvariableopJsavev2_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_read_readvariableop>savev2_simple_rnn_7_simple_rnn_cell_7_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableopGsavev2_adam_simple_rnn_6_simple_rnn_cell_6_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_6_simple_rnn_cell_6_bias_m_read_readvariableopGsavev2_adam_simple_rnn_7_simple_rnn_cell_7_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_7_simple_rnn_cell_7_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopGsavev2_adam_simple_rnn_6_simple_rnn_cell_6_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_6_simple_rnn_cell_6_bias_v_read_readvariableopGsavev2_adam_simple_rnn_7_simple_rnn_cell_7_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_7_simple_rnn_cell_7_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_7_simple_rnn_cell_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?!
?
while_body_11017699
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_166_11017721_0:P2
$while_simple_rnn_cell_166_11017723_0:P6
$while_simple_rnn_cell_166_11017725_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_166_11017721:P0
"while_simple_rnn_cell_166_11017723:P4
"while_simple_rnn_cell_166_11017725:PP??1while/simple_rnn_cell_166/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/simple_rnn_cell_166/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_166_11017721_0$while_simple_rnn_cell_166_11017723_0$while_simple_rnn_cell_166_11017725_0*
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
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11017686?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_166/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_166/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp2^while/simple_rnn_cell_166/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_166_11017721$while_simple_rnn_cell_166_11017721_0"J
"while_simple_rnn_cell_166_11017723$while_simple_rnn_cell_166_11017723_0"J
"while_simple_rnn_cell_166_11017725$while_simple_rnn_cell_166_11017725_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2f
1while/simple_rnn_cell_166/StatefulPartitionedCall1while/simple_rnn_cell_166/StatefulPartitionedCall: 
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
/__inference_sequential_3_layer_call_fn_11019011

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
J__inference_sequential_3_layer_call_and_return_conditional_losses_11018873o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?>
?
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11020143
inputs_0D
2simple_rnn_cell_167_matmul_readvariableop_resource:PdA
3simple_rnn_cell_167_biasadd_readvariableop_resource:dF
4simple_rnn_cell_167_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_167/BiasAdd/ReadVariableOp?)simple_rnn_cell_167/MatMul/ReadVariableOp?+simple_rnn_cell_167/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_167_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_167/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_167_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_167/BiasAddBiasAdd$simple_rnn_cell_167/MatMul:product:02simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_167_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_167/MatMul_1MatMulzeros:output:03simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_167/addAddV2$simple_rnn_cell_167/BiasAdd:output:0&simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_167/TanhTanhsimple_rnn_cell_167/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_167_matmul_readvariableop_resource3simple_rnn_cell_167_biasadd_readvariableop_resource4simple_rnn_cell_167_matmul_1_readvariableop_resource*
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
while_body_11020077*
condR
while_cond_11020076*8
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
NoOpNoOp+^simple_rnn_cell_167/BiasAdd/ReadVariableOp*^simple_rnn_cell_167/MatMul/ReadVariableOp,^simple_rnn_cell_167/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2X
*simple_rnn_cell_167/BiasAdd/ReadVariableOp*simple_rnn_cell_167/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_167/MatMul/ReadVariableOp)simple_rnn_cell_167/MatMul/ReadVariableOp2Z
+simple_rnn_cell_167/MatMul_1/ReadVariableOp+simple_rnn_cell_167/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?	
f
G__inference_dropout_7_layer_call_and_return_conditional_losses_11020494

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
?-
?
while_body_11020077
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_167_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_167_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_167_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_167_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_167/MatMul/ReadVariableOp?1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_167_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_167/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_167/BiasAddBiasAdd*while/simple_rnn_cell_167/MatMul:product:08while/simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_167/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_167/addAddV2*while/simple_rnn_cell_167/BiasAdd:output:0,while/simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_167/TanhTanh!while/simple_rnn_cell_167/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_167/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_167/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_167/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_167/MatMul/ReadVariableOp2^while/simple_rnn_cell_167/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_167_biasadd_readvariableop_resource;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_167_matmul_1_readvariableop_resource<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_167_matmul_readvariableop_resource:while_simple_rnn_cell_167_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_167/MatMul/ReadVariableOp/while/simple_rnn_cell_167/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp: 
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
while_body_11018392
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_167_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_167_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_167_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_167_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_167/MatMul/ReadVariableOp?1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_167_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_167/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_167/BiasAddBiasAdd*while/simple_rnn_cell_167/MatMul:product:08while/simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_167/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_167/addAddV2*while/simple_rnn_cell_167/BiasAdd:output:0,while/simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_167/TanhTanh!while/simple_rnn_cell_167/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_167/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_167/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_167/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_167/MatMul/ReadVariableOp2^while/simple_rnn_cell_167/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_167_biasadd_readvariableop_resource;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_167_matmul_1_readvariableop_resource<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_167_matmul_readvariableop_resource:while_simple_rnn_cell_167_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_167/MatMul/ReadVariableOp/while/simple_rnn_cell_167/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp: 
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
/__inference_simple_rnn_6_layer_call_fn_11019521

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
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11018336s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
while_body_11019682
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_166_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_166_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_166_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_166_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_166/MatMul/ReadVariableOp?1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_166_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_166/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_166/BiasAddBiasAdd*while/simple_rnn_cell_166/MatMul:product:08while/simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_166/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_166/addAddV2*while/simple_rnn_cell_166/BiasAdd:output:0,while/simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_166/TanhTanh!while/simple_rnn_cell_166/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_166/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_166/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_166/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_166/MatMul/ReadVariableOp2^while/simple_rnn_cell_166/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_166_biasadd_readvariableop_resource;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_166_matmul_1_readvariableop_resource<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_166_matmul_readvariableop_resource:while_simple_rnn_cell_166_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_166/MatMul/ReadVariableOp/while/simple_rnn_cell_166/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp: 
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
while_body_11018597
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_167_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_167_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_167_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_167_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_167/MatMul/ReadVariableOp?1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_167_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_167/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_167/BiasAddBiasAdd*while/simple_rnn_cell_167/MatMul:product:08while/simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_167/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_167/addAddV2*while/simple_rnn_cell_167/BiasAdd:output:0,while/simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_167/TanhTanh!while/simple_rnn_cell_167/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_167/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_167/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_167/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_167/MatMul/ReadVariableOp2^while/simple_rnn_cell_167/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_167_biasadd_readvariableop_resource;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_167_matmul_1_readvariableop_resource<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_167_matmul_readvariableop_resource:while_simple_rnn_cell_167_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_167/MatMul/ReadVariableOp/while/simple_rnn_cell_167/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp: 
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
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11020467

inputsD
2simple_rnn_cell_167_matmul_readvariableop_resource:PdA
3simple_rnn_cell_167_biasadd_readvariableop_resource:dF
4simple_rnn_cell_167_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_167/BiasAdd/ReadVariableOp?)simple_rnn_cell_167/MatMul/ReadVariableOp?+simple_rnn_cell_167/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_167_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_167/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_167_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_167/BiasAddBiasAdd$simple_rnn_cell_167/MatMul:product:02simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_167_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_167/MatMul_1MatMulzeros:output:03simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_167/addAddV2$simple_rnn_cell_167/BiasAdd:output:0&simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_167/TanhTanhsimple_rnn_cell_167/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_167_matmul_readvariableop_resource3simple_rnn_cell_167_biasadd_readvariableop_resource4simple_rnn_cell_167_matmul_1_readvariableop_resource*
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
while_body_11020401*
condR
while_cond_11020400*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_167/BiasAdd/ReadVariableOp*^simple_rnn_cell_167/MatMul/ReadVariableOp,^simple_rnn_cell_167/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_167/BiasAdd/ReadVariableOp*simple_rnn_cell_167/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_167/MatMul/ReadVariableOp)simple_rnn_cell_167/MatMul/ReadVariableOp2Z
+simple_rnn_cell_167/MatMul_1/ReadVariableOp+simple_rnn_cell_167/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?-
?
while_body_11018270
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_166_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_166_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_166_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_166_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_166/MatMul/ReadVariableOp?1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_166_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_166/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_166/BiasAddBiasAdd*while/simple_rnn_cell_166/MatMul:product:08while/simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_166/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_166/addAddV2*while/simple_rnn_cell_166/BiasAdd:output:0,while/simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_166/TanhTanh!while/simple_rnn_cell_166/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_166/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_166/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_166/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_166/MatMul/ReadVariableOp2^while/simple_rnn_cell_166/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_166_biasadd_readvariableop_resource;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_166_matmul_1_readvariableop_resource<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_166_matmul_readvariableop_resource:while_simple_rnn_cell_166_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_166/MatMul/ReadVariableOp/while/simple_rnn_cell_166/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp: 
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

?
 simple_rnn_7_while_cond_110191576
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_28
4simple_rnn_7_while_less_simple_rnn_7_strided_slice_1P
Lsimple_rnn_7_while_simple_rnn_7_while_cond_11019157___redundant_placeholder0P
Lsimple_rnn_7_while_simple_rnn_7_while_cond_11019157___redundant_placeholder1P
Lsimple_rnn_7_while_simple_rnn_7_while_cond_11019157___redundant_placeholder2P
Lsimple_rnn_7_while_simple_rnn_7_while_cond_11019157___redundant_placeholder3
simple_rnn_7_while_identity
?
simple_rnn_7/while/LessLesssimple_rnn_7_while_placeholder4simple_rnn_7_while_less_simple_rnn_7_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0*(
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
?
?
/__inference_simple_rnn_6_layer_call_fn_11019510
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
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11017921|
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
?
?
while_cond_11019789
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11019789___redundant_placeholder06
2while_while_cond_11019789___redundant_placeholder16
2while_while_cond_11019789___redundant_placeholder26
2while_while_cond_11019789___redundant_placeholder3
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

f
G__inference_dropout_6_layer_call_and_return_conditional_losses_11018692

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
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
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
:?????????Ps
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Pm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?!
?
while_body_11018150
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_167_11018172_0:Pd2
$while_simple_rnn_cell_167_11018174_0:d6
$while_simple_rnn_cell_167_11018176_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_167_11018172:Pd0
"while_simple_rnn_cell_167_11018174:d4
"while_simple_rnn_cell_167_11018176:dd??1while/simple_rnn_cell_167/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
1while/simple_rnn_cell_167/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_167_11018172_0$while_simple_rnn_cell_167_11018174_0$while_simple_rnn_cell_167_11018176_0*
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
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11018098?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_167/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_167/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp2^while/simple_rnn_cell_167/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_167_11018172$while_simple_rnn_cell_167_11018172_0"J
"while_simple_rnn_cell_167_11018174$while_simple_rnn_cell_167_11018174_0"J
"while_simple_rnn_cell_167_11018176$while_simple_rnn_cell_167_11018176_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2f
1while/simple_rnn_cell_167/StatefulPartitionedCall1while/simple_rnn_cell_167/StatefulPartitionedCall: 
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
while_cond_11018149
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11018149___redundant_placeholder06
2while_while_cond_11018149___redundant_placeholder16
2while_while_cond_11018149___redundant_placeholder26
2while_while_cond_11018149___redundant_placeholder3
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
while_body_11019574
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_166_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_166_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_166_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_166_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_166/MatMul/ReadVariableOp?1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_166_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_166/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_166/BiasAddBiasAdd*while/simple_rnn_cell_166/MatMul:product:08while/simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_166/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_166/addAddV2*while/simple_rnn_cell_166/BiasAdd:output:0,while/simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_166/TanhTanh!while/simple_rnn_cell_166/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_166/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_166/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_166/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_166/MatMul/ReadVariableOp2^while/simple_rnn_cell_166/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_166_biasadd_readvariableop_resource;while_simple_rnn_cell_166_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_166_matmul_1_readvariableop_resource<while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_166_matmul_readvariableop_resource:while_simple_rnn_cell_166_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp0while/simple_rnn_cell_166/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_166/MatMul/ReadVariableOp/while/simple_rnn_cell_166/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp1while/simple_rnn_cell_166/MatMul_1/ReadVariableOp: 
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
?4
?
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11017762

inputs.
simple_rnn_cell_166_11017687:P*
simple_rnn_cell_166_11017689:P.
simple_rnn_cell_166_11017691:PP
identity??+simple_rnn_cell_166/StatefulPartitionedCall?while;
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
+simple_rnn_cell_166/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_166_11017687simple_rnn_cell_166_11017689simple_rnn_cell_166_11017691*
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
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11017686n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_166_11017687simple_rnn_cell_166_11017689simple_rnn_cell_166_11017691*
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
while_body_11017699*
condR
while_cond_11017698*8
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
NoOpNoOp,^simple_rnn_cell_166/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+simple_rnn_cell_166/StatefulPartitionedCall+simple_rnn_cell_166/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
-sequential_3_simple_rnn_7_while_cond_11017564P
Lsequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_while_loop_counterV
Rsequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_while_maximum_iterations/
+sequential_3_simple_rnn_7_while_placeholder1
-sequential_3_simple_rnn_7_while_placeholder_11
-sequential_3_simple_rnn_7_while_placeholder_2R
Nsequential_3_simple_rnn_7_while_less_sequential_3_simple_rnn_7_strided_slice_1j
fsequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_while_cond_11017564___redundant_placeholder0j
fsequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_while_cond_11017564___redundant_placeholder1j
fsequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_while_cond_11017564___redundant_placeholder2j
fsequential_3_simple_rnn_7_while_sequential_3_simple_rnn_7_while_cond_11017564___redundant_placeholder3,
(sequential_3_simple_rnn_7_while_identity
?
$sequential_3/simple_rnn_7/while/LessLess+sequential_3_simple_rnn_7_while_placeholderNsequential_3_simple_rnn_7_while_less_sequential_3_simple_rnn_7_strided_slice_1*
T0*
_output_shapes
: 
(sequential_3/simple_rnn_7/while/IdentityIdentity(sequential_3/simple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: "]
(sequential_3_simple_rnn_7_while_identity1sequential_3/simple_rnn_7/while/Identity:output:0*(
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
while_cond_11018269
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11018269___redundant_placeholder06
2while_while_cond_11018269___redundant_placeholder16
2while_while_cond_11018269___redundant_placeholder26
2while_while_cond_11018269___redundant_placeholder3
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
?
?
/__inference_simple_rnn_7_layer_call_fn_11020013
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
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11018213o
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
?
/__inference_sequential_3_layer_call_fn_11018990

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
J__inference_sequential_3_layer_call_and_return_conditional_losses_11018490o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?=
?
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11020359

inputsD
2simple_rnn_cell_167_matmul_readvariableop_resource:PdA
3simple_rnn_cell_167_biasadd_readvariableop_resource:dF
4simple_rnn_cell_167_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_167/BiasAdd/ReadVariableOp?)simple_rnn_cell_167/MatMul/ReadVariableOp?+simple_rnn_cell_167/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_167_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_167/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_167_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_167/BiasAddBiasAdd$simple_rnn_cell_167/MatMul:product:02simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_167_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_167/MatMul_1MatMulzeros:output:03simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_167/addAddV2$simple_rnn_cell_167/BiasAdd:output:0&simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_167/TanhTanhsimple_rnn_cell_167/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_167_matmul_readvariableop_resource3simple_rnn_cell_167_biasadd_readvariableop_resource4simple_rnn_cell_167_matmul_1_readvariableop_resource*
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
while_body_11020293*
condR
while_cond_11020292*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_167/BiasAdd/ReadVariableOp*^simple_rnn_cell_167/MatMul/ReadVariableOp,^simple_rnn_cell_167/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_167/BiasAdd/ReadVariableOp*simple_rnn_cell_167/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_167/MatMul/ReadVariableOp)simple_rnn_cell_167/MatMul/ReadVariableOp2Z
+simple_rnn_cell_167/MatMul_1/ReadVariableOp+simple_rnn_cell_167/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?-
?
while_body_11020293
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_167_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_167_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_167_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_167_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_167/MatMul/ReadVariableOp?1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_167_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_167/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_167/BiasAddBiasAdd*while/simple_rnn_cell_167/MatMul:product:08while/simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_167/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_167/addAddV2*while/simple_rnn_cell_167/BiasAdd:output:0,while/simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_167/TanhTanh!while/simple_rnn_cell_167/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_167/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_167/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_167/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_167/MatMul/ReadVariableOp2^while/simple_rnn_cell_167/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_167_biasadd_readvariableop_resource;while_simple_rnn_cell_167_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_167_matmul_1_readvariableop_resource<while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_167_matmul_readvariableop_resource:while_simple_rnn_cell_167_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp0while/simple_rnn_cell_167/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_167/MatMul/ReadVariableOp/while/simple_rnn_cell_167/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp1while/simple_rnn_cell_167/MatMul_1/ReadVariableOp: 
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
while_cond_11018749
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11018749___redundant_placeholder06
2while_while_cond_11018749___redundant_placeholder16
2while_while_cond_11018749___redundant_placeholder26
2while_while_cond_11018749___redundant_placeholder3
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
?9
?
 simple_rnn_6_while_body_110190536
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_25
1simple_rnn_6_while_simple_rnn_6_strided_slice_1_0q
msimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resource_0:PV
Hsimple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resource_0:P[
Isimple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0:PP
simple_rnn_6_while_identity!
simple_rnn_6_while_identity_1!
simple_rnn_6_while_identity_2!
simple_rnn_6_while_identity_3!
simple_rnn_6_while_identity_43
/simple_rnn_6_while_simple_rnn_6_strided_slice_1o
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resource:PT
Fsimple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resource:PY
Gsimple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resource:PP??=simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOp?<simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOp?>simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp?
Dsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
6simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_6_while_placeholderMsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
<simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
-simple_rnn_6/while/simple_rnn_cell_166/MatMulMatMul=simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
=simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
.simple_rnn_6/while/simple_rnn_cell_166/BiasAddBiasAdd7simple_rnn_6/while/simple_rnn_cell_166/MatMul:product:0Esimple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
>simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
/simple_rnn_6/while/simple_rnn_cell_166/MatMul_1MatMul simple_rnn_6_while_placeholder_2Fsimple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_6/while/simple_rnn_cell_166/addAddV27simple_rnn_6/while/simple_rnn_cell_166/BiasAdd:output:09simple_rnn_6/while/simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_6/while/simple_rnn_cell_166/TanhTanh.simple_rnn_6/while/simple_rnn_cell_166/add:z:0*
T0*'
_output_shapes
:?????????P?
7simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_6_while_placeholder_1simple_rnn_6_while_placeholder/simple_rnn_6/while/simple_rnn_cell_166/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_6/while/addAddV2simple_rnn_6_while_placeholder!simple_rnn_6/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_6/while/add_1AddV22simple_rnn_6_while_simple_rnn_6_while_loop_counter#simple_rnn_6/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_6/while/IdentityIdentitysimple_rnn_6/while/add_1:z:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_6/while/Identity_1Identity8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_6/while/Identity_2Identitysimple_rnn_6/while/add:z:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_6/while/Identity_3IdentityGsimple_rnn_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_6/while/Identity_4Identity/simple_rnn_6/while/simple_rnn_cell_166/Tanh:y:0^simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:?????????P?
simple_rnn_6/while/NoOpNoOp>^simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOp=^simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOp?^simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_6_while_identity$simple_rnn_6/while/Identity:output:0"G
simple_rnn_6_while_identity_1&simple_rnn_6/while/Identity_1:output:0"G
simple_rnn_6_while_identity_2&simple_rnn_6/while/Identity_2:output:0"G
simple_rnn_6_while_identity_3&simple_rnn_6/while/Identity_3:output:0"G
simple_rnn_6_while_identity_4&simple_rnn_6/while/Identity_4:output:0"d
/simple_rnn_6_while_simple_rnn_6_strided_slice_11simple_rnn_6_while_simple_rnn_6_strided_slice_1_0"?
Fsimple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resourceHsimple_rnn_6_while_simple_rnn_cell_166_biasadd_readvariableop_resource_0"?
Gsimple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resourceIsimple_rnn_6_while_simple_rnn_cell_166_matmul_1_readvariableop_resource_0"?
Esimple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resourceGsimple_rnn_6_while_simple_rnn_cell_166_matmul_readvariableop_resource_0"?
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensormsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2~
=simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOp=simple_rnn_6/while/simple_rnn_cell_166/BiasAdd/ReadVariableOp2|
<simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOp<simple_rnn_6/while/simple_rnn_cell_166/MatMul/ReadVariableOp2?
>simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp>simple_rnn_6/while/simple_rnn_cell_166/MatMul_1/ReadVariableOp: 
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
?
?
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11020637

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
?	
?
E__inference_dense_3_layer_call_and_return_conditional_losses_11018483

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
?
?
/__inference_simple_rnn_6_layer_call_fn_11019499
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
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11017762|
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
?

?
 simple_rnn_6_while_cond_110192726
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_28
4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1P
Lsimple_rnn_6_while_simple_rnn_6_while_cond_11019272___redundant_placeholder0P
Lsimple_rnn_6_while_simple_rnn_6_while_cond_11019272___redundant_placeholder1P
Lsimple_rnn_6_while_simple_rnn_6_while_cond_11019272___redundant_placeholder2P
Lsimple_rnn_6_while_simple_rnn_6_while_cond_11019272___redundant_placeholder3
simple_rnn_6_while_identity
?
simple_rnn_6/while/LessLesssimple_rnn_6_while_placeholder4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_6/while/IdentityIdentitysimple_rnn_6/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_6_while_identity$simple_rnn_6/while/Identity:output:0*(
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
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11017686

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
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11017921

inputs.
simple_rnn_cell_166_11017846:P*
simple_rnn_cell_166_11017848:P.
simple_rnn_cell_166_11017850:PP
identity??+simple_rnn_cell_166/StatefulPartitionedCall?while;
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
+simple_rnn_cell_166/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_166_11017846simple_rnn_cell_166_11017848simple_rnn_cell_166_11017850*
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
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11017806n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_166_11017846simple_rnn_cell_166_11017848simple_rnn_cell_166_11017850*
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
while_body_11017858*
condR
while_cond_11017857*8
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
NoOpNoOp,^simple_rnn_cell_166/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+simple_rnn_cell_166/StatefulPartitionedCall+simple_rnn_cell_166/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?=
?
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11019964

inputsD
2simple_rnn_cell_166_matmul_readvariableop_resource:PA
3simple_rnn_cell_166_biasadd_readvariableop_resource:PF
4simple_rnn_cell_166_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_166/BiasAdd/ReadVariableOp?)simple_rnn_cell_166/MatMul/ReadVariableOp?+simple_rnn_cell_166/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_166_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_166/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_166_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_166/BiasAddBiasAdd$simple_rnn_cell_166/MatMul:product:02simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_166_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_166/MatMul_1MatMulzeros:output:03simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_166/addAddV2$simple_rnn_cell_166/BiasAdd:output:0&simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_166/TanhTanhsimple_rnn_cell_166/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_166_matmul_readvariableop_resource3simple_rnn_cell_166_biasadd_readvariableop_resource4simple_rnn_cell_166_matmul_1_readvariableop_resource*
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
while_body_11019898*
condR
while_cond_11019897*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_166/BiasAdd/ReadVariableOp*^simple_rnn_cell_166/MatMul/ReadVariableOp,^simple_rnn_cell_166/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_166/BiasAdd/ReadVariableOp*simple_rnn_cell_166/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_166/MatMul/ReadVariableOp)simple_rnn_cell_166/MatMul/ReadVariableOp2Z
+simple_rnn_cell_166/MatMul_1/ReadVariableOp+simple_rnn_cell_166/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11020575

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
?

?
 simple_rnn_6_while_cond_110190526
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_28
4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1P
Lsimple_rnn_6_while_simple_rnn_6_while_cond_11019052___redundant_placeholder0P
Lsimple_rnn_6_while_simple_rnn_6_while_cond_11019052___redundant_placeholder1P
Lsimple_rnn_6_while_simple_rnn_6_while_cond_11019052___redundant_placeholder2P
Lsimple_rnn_6_while_simple_rnn_6_while_cond_11019052___redundant_placeholder3
simple_rnn_6_while_identity
?
simple_rnn_6/while/LessLesssimple_rnn_6_while_placeholder4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_6/while/IdentityIdentitysimple_rnn_6/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_6_while_identity$simple_rnn_6/while/Identity:output:0*(
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
while_cond_11020292
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11020292___redundant_placeholder06
2while_while_cond_11020292___redundant_placeholder16
2while_while_cond_11020292___redundant_placeholder26
2while_while_cond_11020292___redundant_placeholder3
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
??
?	
J__inference_sequential_3_layer_call_and_return_conditional_losses_11019465

inputsQ
?simple_rnn_6_simple_rnn_cell_166_matmul_readvariableop_resource:PN
@simple_rnn_6_simple_rnn_cell_166_biasadd_readvariableop_resource:PS
Asimple_rnn_6_simple_rnn_cell_166_matmul_1_readvariableop_resource:PPQ
?simple_rnn_7_simple_rnn_cell_167_matmul_readvariableop_resource:PdN
@simple_rnn_7_simple_rnn_cell_167_biasadd_readvariableop_resource:dS
Asimple_rnn_7_simple_rnn_cell_167_matmul_1_readvariableop_resource:dd8
&dense_3_matmul_readvariableop_resource:d5
'dense_3_biasadd_readvariableop_resource:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?7simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOp?6simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOp?8simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp?simple_rnn_6/while?7simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOp?6simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOp?8simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp?simple_rnn_7/whileH
simple_rnn_6/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_6/strided_sliceStridedSlicesimple_rnn_6/Shape:output:0)simple_rnn_6/strided_slice/stack:output:0+simple_rnn_6/strided_slice/stack_1:output:0+simple_rnn_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
simple_rnn_6/zeros/packedPack#simple_rnn_6/strided_slice:output:0$simple_rnn_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_6/zerosFill"simple_rnn_6/zeros/packed:output:0!simple_rnn_6/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pp
simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_6/transpose	Transposeinputs$simple_rnn_6/transpose/perm:output:0*
T0*+
_output_shapes
:?????????^
simple_rnn_6/Shape_1Shapesimple_rnn_6/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_6/strided_slice_1StridedSlicesimple_rnn_6/Shape_1:output:0+simple_rnn_6/strided_slice_1/stack:output:0-simple_rnn_6/strided_slice_1/stack_1:output:0-simple_rnn_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_6/TensorArrayV2TensorListReserve1simple_rnn_6/TensorArrayV2/element_shape:output:0%simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
4simple_rnn_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_6/transpose:y:0Ksimple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_6/strided_slice_2StridedSlicesimple_rnn_6/transpose:y:0+simple_rnn_6/strided_slice_2/stack:output:0-simple_rnn_6/strided_slice_2/stack_1:output:0-simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
6simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOpReadVariableOp?simple_rnn_6_simple_rnn_cell_166_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
'simple_rnn_6/simple_rnn_cell_166/MatMulMatMul%simple_rnn_6/strided_slice_2:output:0>simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
7simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_6_simple_rnn_cell_166_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
(simple_rnn_6/simple_rnn_cell_166/BiasAddBiasAdd1simple_rnn_6/simple_rnn_cell_166/MatMul:product:0?simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_6_simple_rnn_cell_166_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
)simple_rnn_6/simple_rnn_cell_166/MatMul_1MatMulsimple_rnn_6/zeros:output:0@simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
$simple_rnn_6/simple_rnn_cell_166/addAddV21simple_rnn_6/simple_rnn_cell_166/BiasAdd:output:03simple_rnn_6/simple_rnn_cell_166/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
%simple_rnn_6/simple_rnn_cell_166/TanhTanh(simple_rnn_6/simple_rnn_cell_166/add:z:0*
T0*'
_output_shapes
:?????????P{
*simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
simple_rnn_6/TensorArrayV2_1TensorListReserve3simple_rnn_6/TensorArrayV2_1/element_shape:output:0%simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_6/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_6/whileWhile(simple_rnn_6/while/loop_counter:output:0.simple_rnn_6/while/maximum_iterations:output:0simple_rnn_6/time:output:0%simple_rnn_6/TensorArrayV2_1:handle:0simple_rnn_6/zeros:output:0%simple_rnn_6/strided_slice_1:output:0Dsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_6_simple_rnn_cell_166_matmul_readvariableop_resource@simple_rnn_6_simple_rnn_cell_166_biasadd_readvariableop_resourceAsimple_rnn_6_simple_rnn_cell_166_matmul_1_readvariableop_resource*
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
 simple_rnn_6_while_body_11019273*,
cond$R"
 simple_rnn_6_while_cond_11019272*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
=simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_6/while:output:3Fsimple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0u
"simple_rnn_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_6/strided_slice_3StridedSlice8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_6/strided_slice_3/stack:output:0-simple_rnn_6/strided_slice_3/stack_1:output:0-simple_rnn_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maskr
simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_6/transpose_1	Transpose8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P\
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_6/dropout/MulMulsimple_rnn_6/transpose_1:y:0 dropout_6/dropout/Const:output:0*
T0*+
_output_shapes
:?????????Pc
dropout_6/dropout/ShapeShapesimple_rnn_6/transpose_1:y:0*
T0*
_output_shapes
:?
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????P?
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????P?
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
simple_rnn_7/ShapeShapedropout_6/dropout/Mul_1:z:0*
T0*
_output_shapes
:j
 simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_sliceStridedSlicesimple_rnn_7/Shape:output:0)simple_rnn_7/strided_slice/stack:output:0+simple_rnn_7/strided_slice/stack_1:output:0+simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
simple_rnn_7/zeros/packedPack#simple_rnn_7/strided_slice:output:0$simple_rnn_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_7/zerosFill"simple_rnn_7/zeros/packed:output:0!simple_rnn_7/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dp
simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_7/transpose	Transposedropout_6/dropout/Mul_1:z:0$simple_rnn_7/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P^
simple_rnn_7/Shape_1Shapesimple_rnn_7/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_slice_1StridedSlicesimple_rnn_7/Shape_1:output:0+simple_rnn_7/strided_slice_1/stack:output:0-simple_rnn_7/strided_slice_1/stack_1:output:0-simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_7/TensorArrayV2TensorListReserve1simple_rnn_7/TensorArrayV2/element_shape:output:0%simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_7/transpose:y:0Ksimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_slice_2StridedSlicesimple_rnn_7/transpose:y:0+simple_rnn_7/strided_slice_2/stack:output:0-simple_rnn_7/strided_slice_2/stack_1:output:0-simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
6simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_167_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
'simple_rnn_7/simple_rnn_cell_167/MatMulMatMul%simple_rnn_7/strided_slice_2:output:0>simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
7simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_7_simple_rnn_cell_167_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
(simple_rnn_7/simple_rnn_cell_167/BiasAddBiasAdd1simple_rnn_7/simple_rnn_cell_167/MatMul:product:0?simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_7_simple_rnn_cell_167_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
)simple_rnn_7/simple_rnn_cell_167/MatMul_1MatMulsimple_rnn_7/zeros:output:0@simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
$simple_rnn_7/simple_rnn_cell_167/addAddV21simple_rnn_7/simple_rnn_cell_167/BiasAdd:output:03simple_rnn_7/simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
%simple_rnn_7/simple_rnn_cell_167/TanhTanh(simple_rnn_7/simple_rnn_cell_167/add:z:0*
T0*'
_output_shapes
:?????????d{
*simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
simple_rnn_7/TensorArrayV2_1TensorListReserve3simple_rnn_7/TensorArrayV2_1/element_shape:output:0%simple_rnn_7/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_7/whileWhile(simple_rnn_7/while/loop_counter:output:0.simple_rnn_7/while/maximum_iterations:output:0simple_rnn_7/time:output:0%simple_rnn_7/TensorArrayV2_1:handle:0simple_rnn_7/zeros:output:0%simple_rnn_7/strided_slice_1:output:0Dsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_7_simple_rnn_cell_167_matmul_readvariableop_resource@simple_rnn_7_simple_rnn_cell_167_biasadd_readvariableop_resourceAsimple_rnn_7_simple_rnn_cell_167_matmul_1_readvariableop_resource*
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
 simple_rnn_7_while_body_11019385*,
cond$R"
 simple_rnn_7_while_cond_11019384*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_7/while:output:3Fsimple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0u
"simple_rnn_7/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_7/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_7/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_7/strided_slice_3StridedSlice8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_7/strided_slice_3/stack:output:0-simple_rnn_7/strided_slice_3/stack_1:output:0-simple_rnn_7/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maskr
simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_7/transpose_1	Transpose8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_7/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d\
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_7/dropout/MulMul%simple_rnn_7/strided_slice_3:output:0 dropout_7/dropout/Const:output:0*
T0*'
_output_shapes
:?????????dl
dropout_7/dropout/ShapeShape%simple_rnn_7/strided_slice_3:output:0*
T0*
_output_shapes
:?
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_3/MatMulMatMuldropout_7/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp8^simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOp7^simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOp9^simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp^simple_rnn_6/while8^simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOp7^simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOp9^simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp^simple_rnn_7/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2r
7simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOp7simple_rnn_6/simple_rnn_cell_166/BiasAdd/ReadVariableOp2p
6simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOp6simple_rnn_6/simple_rnn_cell_166/MatMul/ReadVariableOp2t
8simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp8simple_rnn_6/simple_rnn_cell_166/MatMul_1/ReadVariableOp2(
simple_rnn_6/whilesimple_rnn_6/while2r
7simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOp7simple_rnn_7/simple_rnn_cell_167/BiasAdd/ReadVariableOp2p
6simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOp6simple_rnn_7/simple_rnn_cell_167/MatMul/ReadVariableOp2t
8simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp8simple_rnn_7/simple_rnn_cell_167/MatMul_1/ReadVariableOp2(
simple_rnn_7/whilesimple_rnn_7/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
6__inference_simple_rnn_cell_166_layer_call_fn_11020541

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
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11017806o
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
?
?
while_cond_11017857
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11017857___redundant_placeholder06
2while_while_cond_11017857___redundant_placeholder16
2while_while_cond_11017857___redundant_placeholder26
2while_while_cond_11017857___redundant_placeholder3
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
?
H
,__inference_dropout_6_layer_call_fn_11019969

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
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_6_layer_call_and_return_conditional_losses_11018349d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
while_cond_11017990
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11017990___redundant_placeholder06
2while_while_cond_11017990___redundant_placeholder16
2while_while_cond_11017990___redundant_placeholder26
2while_while_cond_11017990___redundant_placeholder3
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
?
e
G__inference_dropout_6_layer_call_and_return_conditional_losses_11019979

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?!
?
while_body_11017858
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_166_11017880_0:P2
$while_simple_rnn_cell_166_11017882_0:P6
$while_simple_rnn_cell_166_11017884_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_166_11017880:P0
"while_simple_rnn_cell_166_11017882:P4
"while_simple_rnn_cell_166_11017884:PP??1while/simple_rnn_cell_166/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/simple_rnn_cell_166/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_166_11017880_0$while_simple_rnn_cell_166_11017882_0$while_simple_rnn_cell_166_11017884_0*
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
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11017806?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_166/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_166/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp2^while/simple_rnn_cell_166/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_166_11017880$while_simple_rnn_cell_166_11017880_0"J
"while_simple_rnn_cell_166_11017882$while_simple_rnn_cell_166_11017882_0"J
"while_simple_rnn_cell_166_11017884$while_simple_rnn_cell_166_11017884_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2f
1while/simple_rnn_cell_166/StatefulPartitionedCall1while/simple_rnn_cell_166/StatefulPartitionedCall: 
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
&__inference_signature_wrapper_11019488
simple_rnn_6_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_11017638o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_6_input
?
?
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11020620

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
?

?
6__inference_simple_rnn_cell_167_layer_call_fn_11020603

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
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11018098o
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
?=
?
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11018458

inputsD
2simple_rnn_cell_167_matmul_readvariableop_resource:PdA
3simple_rnn_cell_167_biasadd_readvariableop_resource:dF
4simple_rnn_cell_167_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_167/BiasAdd/ReadVariableOp?)simple_rnn_cell_167/MatMul/ReadVariableOp?+simple_rnn_cell_167/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_167_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_167/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_167_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_167/BiasAddBiasAdd$simple_rnn_cell_167/MatMul:product:02simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_167_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_167/MatMul_1MatMulzeros:output:03simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_167/addAddV2$simple_rnn_cell_167/BiasAdd:output:0&simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_167/TanhTanhsimple_rnn_cell_167/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_167_matmul_readvariableop_resource3simple_rnn_cell_167_biasadd_readvariableop_resource4simple_rnn_cell_167_matmul_1_readvariableop_resource*
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
while_body_11018392*
condR
while_cond_11018391*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_167/BiasAdd/ReadVariableOp*^simple_rnn_cell_167/MatMul/ReadVariableOp,^simple_rnn_cell_167/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_167/BiasAdd/ReadVariableOp*simple_rnn_cell_167/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_167/MatMul/ReadVariableOp)simple_rnn_cell_167/MatMul/ReadVariableOp2Z
+simple_rnn_cell_167/MatMul_1/ReadVariableOp+simple_rnn_cell_167/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?	
f
G__inference_dropout_7_layer_call_and_return_conditional_losses_11018539

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
?
?
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11020558

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
?9
?
 simple_rnn_7_while_body_110191586
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_25
1simple_rnn_7_while_simple_rnn_7_strided_slice_1_0q
msimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resource_0:PdV
Hsimple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resource_0:d[
Isimple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0:dd
simple_rnn_7_while_identity!
simple_rnn_7_while_identity_1!
simple_rnn_7_while_identity_2!
simple_rnn_7_while_identity_3!
simple_rnn_7_while_identity_43
/simple_rnn_7_while_simple_rnn_7_strided_slice_1o
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resource:PdT
Fsimple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resource:dY
Gsimple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resource:dd??=simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOp?<simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOp?>simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp?
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_7_while_placeholderMsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
<simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
-simple_rnn_7/while/simple_rnn_cell_167/MatMulMatMul=simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
=simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
.simple_rnn_7/while/simple_rnn_cell_167/BiasAddBiasAdd7simple_rnn_7/while/simple_rnn_cell_167/MatMul:product:0Esimple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
>simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
/simple_rnn_7/while/simple_rnn_cell_167/MatMul_1MatMul simple_rnn_7_while_placeholder_2Fsimple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_7/while/simple_rnn_cell_167/addAddV27simple_rnn_7/while/simple_rnn_cell_167/BiasAdd:output:09simple_rnn_7/while/simple_rnn_cell_167/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_7/while/simple_rnn_cell_167/TanhTanh.simple_rnn_7/while/simple_rnn_cell_167/add:z:0*
T0*'
_output_shapes
:?????????d?
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_7_while_placeholder_1simple_rnn_7_while_placeholder/simple_rnn_7/while/simple_rnn_cell_167/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_7/while/addAddV2simple_rnn_7_while_placeholder!simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_7/while/add_1AddV22simple_rnn_7_while_simple_rnn_7_while_loop_counter#simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/add_1:z:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_1Identity8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_2Identitysimple_rnn_7/while/add:z:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_7/while/Identity_3IdentityGsimple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_7/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_7/while/Identity_4Identity/simple_rnn_7/while/simple_rnn_cell_167/Tanh:y:0^simple_rnn_7/while/NoOp*
T0*'
_output_shapes
:?????????d?
simple_rnn_7/while/NoOpNoOp>^simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOp=^simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOp?^simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0"G
simple_rnn_7_while_identity_1&simple_rnn_7/while/Identity_1:output:0"G
simple_rnn_7_while_identity_2&simple_rnn_7/while/Identity_2:output:0"G
simple_rnn_7_while_identity_3&simple_rnn_7/while/Identity_3:output:0"G
simple_rnn_7_while_identity_4&simple_rnn_7/while/Identity_4:output:0"d
/simple_rnn_7_while_simple_rnn_7_strided_slice_11simple_rnn_7_while_simple_rnn_7_strided_slice_1_0"?
Fsimple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resourceHsimple_rnn_7_while_simple_rnn_cell_167_biasadd_readvariableop_resource_0"?
Gsimple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resourceIsimple_rnn_7_while_simple_rnn_cell_167_matmul_1_readvariableop_resource_0"?
Esimple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resourceGsimple_rnn_7_while_simple_rnn_cell_167_matmul_readvariableop_resource_0"?
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensormsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2~
=simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOp=simple_rnn_7/while/simple_rnn_cell_167/BiasAdd/ReadVariableOp2|
<simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOp<simple_rnn_7/while/simple_rnn_cell_167/MatMul/ReadVariableOp2?
>simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp>simple_rnn_7/while/simple_rnn_cell_167/MatMul_1/ReadVariableOp: 
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
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
simple_rnn_6_input?
$serving_default_simple_rnn_6_input:0?????????;
dense_30
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
/__inference_sequential_3_layer_call_fn_11018509
/__inference_sequential_3_layer_call_fn_11018990
/__inference_sequential_3_layer_call_fn_11019011
/__inference_sequential_3_layer_call_fn_11018913?
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_11019231
J__inference_sequential_3_layer_call_and_return_conditional_losses_11019465
J__inference_sequential_3_layer_call_and_return_conditional_losses_11018938
J__inference_sequential_3_layer_call_and_return_conditional_losses_11018963?
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
#__inference__wrapped_model_11017638simple_rnn_6_input"?
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
/__inference_simple_rnn_6_layer_call_fn_11019499
/__inference_simple_rnn_6_layer_call_fn_11019510
/__inference_simple_rnn_6_layer_call_fn_11019521
/__inference_simple_rnn_6_layer_call_fn_11019532?
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
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11019640
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11019748
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11019856
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11019964?
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
,__inference_dropout_6_layer_call_fn_11019969
,__inference_dropout_6_layer_call_fn_11019974?
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
G__inference_dropout_6_layer_call_and_return_conditional_losses_11019979
G__inference_dropout_6_layer_call_and_return_conditional_losses_11019991?
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
/__inference_simple_rnn_7_layer_call_fn_11020002
/__inference_simple_rnn_7_layer_call_fn_11020013
/__inference_simple_rnn_7_layer_call_fn_11020024
/__inference_simple_rnn_7_layer_call_fn_11020035?
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
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11020143
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11020251
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11020359
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11020467?
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
,__inference_dropout_7_layer_call_fn_11020472
,__inference_dropout_7_layer_call_fn_11020477?
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
G__inference_dropout_7_layer_call_and_return_conditional_losses_11020482
G__inference_dropout_7_layer_call_and_return_conditional_losses_11020494?
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
 :d2dense_3/kernel
:2dense_3/bias
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
*__inference_dense_3_layer_call_fn_11020503?
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
E__inference_dense_3_layer_call_and_return_conditional_losses_11020513?
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
&__inference_signature_wrapper_11019488simple_rnn_6_input"?
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
7:5P2%simple_rnn_6/simple_rnn_cell_6/kernel
A:?PP2/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel
1:/P2#simple_rnn_6/simple_rnn_cell_6/bias
7:5Pd2%simple_rnn_7/simple_rnn_cell_7/kernel
A:?dd2/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel
1:/d2#simple_rnn_7/simple_rnn_cell_7/bias
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
6__inference_simple_rnn_cell_166_layer_call_fn_11020527
6__inference_simple_rnn_cell_166_layer_call_fn_11020541?
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
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11020558
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11020575?
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
6__inference_simple_rnn_cell_167_layer_call_fn_11020589
6__inference_simple_rnn_cell_167_layer_call_fn_11020603?
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
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11020620
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11020637?
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
%:#d2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
<::P2,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m
F:DPP26Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m
6:4P2*Adam/simple_rnn_6/simple_rnn_cell_6/bias/m
<::Pd2,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/m
F:Ddd26Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/m
6:4d2*Adam/simple_rnn_7/simple_rnn_cell_7/bias/m
%:#d2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
<::P2,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v
F:DPP26Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v
6:4P2*Adam/simple_rnn_6/simple_rnn_cell_6/bias/v
<::Pd2,Adam/simple_rnn_7/simple_rnn_cell_7/kernel/v
F:Ddd26Adam/simple_rnn_7/simple_rnn_cell_7/recurrent_kernel/v
6:4d2*Adam/simple_rnn_7/simple_rnn_cell_7/bias/v?
#__inference__wrapped_model_11017638~ACBDFE23??<
5?2
0?-
simple_rnn_6_input?????????
? "1?.
,
dense_3!?
dense_3??????????
E__inference_dense_3_layer_call_and_return_conditional_losses_11020513\23/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? }
*__inference_dense_3_layer_call_fn_11020503O23/?,
%?"
 ?
inputs?????????d
? "???????????
G__inference_dropout_6_layer_call_and_return_conditional_losses_11019979d7?4
-?*
$?!
inputs?????????P
p 
? ")?&
?
0?????????P
? ?
G__inference_dropout_6_layer_call_and_return_conditional_losses_11019991d7?4
-?*
$?!
inputs?????????P
p
? ")?&
?
0?????????P
? ?
,__inference_dropout_6_layer_call_fn_11019969W7?4
-?*
$?!
inputs?????????P
p 
? "??????????P?
,__inference_dropout_6_layer_call_fn_11019974W7?4
-?*
$?!
inputs?????????P
p
? "??????????P?
G__inference_dropout_7_layer_call_and_return_conditional_losses_11020482\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? ?
G__inference_dropout_7_layer_call_and_return_conditional_losses_11020494\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? 
,__inference_dropout_7_layer_call_fn_11020472O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d
,__inference_dropout_7_layer_call_fn_11020477O3?0
)?&
 ?
inputs?????????d
p
? "??????????d?
J__inference_sequential_3_layer_call_and_return_conditional_losses_11018938zACBDFE23G?D
=?:
0?-
simple_rnn_6_input?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_11018963zACBDFE23G?D
=?:
0?-
simple_rnn_6_input?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_11019231nACBDFE23;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_11019465nACBDFE23;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_3_layer_call_fn_11018509mACBDFE23G?D
=?:
0?-
simple_rnn_6_input?????????
p 

 
? "???????????
/__inference_sequential_3_layer_call_fn_11018913mACBDFE23G?D
=?:
0?-
simple_rnn_6_input?????????
p

 
? "???????????
/__inference_sequential_3_layer_call_fn_11018990aACBDFE23;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
/__inference_sequential_3_layer_call_fn_11019011aACBDFE23;?8
1?.
$?!
inputs?????????
p

 
? "???????????
&__inference_signature_wrapper_11019488?ACBDFE23U?R
? 
K?H
F
simple_rnn_6_input0?-
simple_rnn_6_input?????????"1?.
,
dense_3!?
dense_3??????????
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11019640?ACBO?L
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
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11019748?ACBO?L
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
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11019856qACB??<
5?2
$?!
inputs?????????

 
p 

 
? ")?&
?
0?????????P
? ?
J__inference_simple_rnn_6_layer_call_and_return_conditional_losses_11019964qACB??<
5?2
$?!
inputs?????????

 
p

 
? ")?&
?
0?????????P
? ?
/__inference_simple_rnn_6_layer_call_fn_11019499}ACBO?L
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
/__inference_simple_rnn_6_layer_call_fn_11019510}ACBO?L
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
/__inference_simple_rnn_6_layer_call_fn_11019521dACB??<
5?2
$?!
inputs?????????

 
p 

 
? "??????????P?
/__inference_simple_rnn_6_layer_call_fn_11019532dACB??<
5?2
$?!
inputs?????????

 
p

 
? "??????????P?
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11020143}DFEO?L
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
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11020251}DFEO?L
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
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11020359mDFE??<
5?2
$?!
inputs?????????P

 
p 

 
? "%?"
?
0?????????d
? ?
J__inference_simple_rnn_7_layer_call_and_return_conditional_losses_11020467mDFE??<
5?2
$?!
inputs?????????P

 
p

 
? "%?"
?
0?????????d
? ?
/__inference_simple_rnn_7_layer_call_fn_11020002pDFEO?L
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
/__inference_simple_rnn_7_layer_call_fn_11020013pDFEO?L
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
/__inference_simple_rnn_7_layer_call_fn_11020024`DFE??<
5?2
$?!
inputs?????????P

 
p 

 
? "??????????d?
/__inference_simple_rnn_7_layer_call_fn_11020035`DFE??<
5?2
$?!
inputs?????????P

 
p

 
? "??????????d?
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11020558?ACB\?Y
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
Q__inference_simple_rnn_cell_166_layer_call_and_return_conditional_losses_11020575?ACB\?Y
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
6__inference_simple_rnn_cell_166_layer_call_fn_11020527?ACB\?Y
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
6__inference_simple_rnn_cell_166_layer_call_fn_11020541?ACB\?Y
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
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11020620?DFE\?Y
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
Q__inference_simple_rnn_cell_167_layer_call_and_return_conditional_losses_11020637?DFE\?Y
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
6__inference_simple_rnn_cell_167_layer_call_fn_11020589?DFE\?Y
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
6__inference_simple_rnn_cell_167_layer_call_fn_11020603?DFE\?Y
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