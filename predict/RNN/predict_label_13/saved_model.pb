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
?"serve*2.8.02v2.8.0-0-g3f878cff5b68??
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:d*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
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
'simple_rnn_18/simple_rnn_cell_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*8
shared_name)'simple_rnn_18/simple_rnn_cell_18/kernel
?
;simple_rnn_18/simple_rnn_cell_18/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_18/simple_rnn_cell_18/kernel*
_output_shapes

:P*
dtype0
?
1simple_rnn_18/simple_rnn_cell_18/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*B
shared_name31simple_rnn_18/simple_rnn_cell_18/recurrent_kernel
?
Esimple_rnn_18/simple_rnn_cell_18/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_18/simple_rnn_cell_18/recurrent_kernel*
_output_shapes

:PP*
dtype0
?
%simple_rnn_18/simple_rnn_cell_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*6
shared_name'%simple_rnn_18/simple_rnn_cell_18/bias
?
9simple_rnn_18/simple_rnn_cell_18/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_18/simple_rnn_cell_18/bias*
_output_shapes
:P*
dtype0
?
'simple_rnn_19/simple_rnn_cell_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*8
shared_name)'simple_rnn_19/simple_rnn_cell_19/kernel
?
;simple_rnn_19/simple_rnn_cell_19/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_19/simple_rnn_cell_19/kernel*
_output_shapes

:Pd*
dtype0
?
1simple_rnn_19/simple_rnn_cell_19/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*B
shared_name31simple_rnn_19/simple_rnn_cell_19/recurrent_kernel
?
Esimple_rnn_19/simple_rnn_cell_19/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_19/simple_rnn_cell_19/recurrent_kernel*
_output_shapes

:dd*
dtype0
?
%simple_rnn_19/simple_rnn_cell_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*6
shared_name'%simple_rnn_19/simple_rnn_cell_19/bias
?
9simple_rnn_19/simple_rnn_cell_19/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_19/simple_rnn_cell_19/bias*
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
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:*
dtype0
?
.Adam/simple_rnn_18/simple_rnn_cell_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*?
shared_name0.Adam/simple_rnn_18/simple_rnn_cell_18/kernel/m
?
BAdam/simple_rnn_18/simple_rnn_cell_18/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_18/simple_rnn_cell_18/kernel/m*
_output_shapes

:P*
dtype0
?
8Adam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*I
shared_name:8Adam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/m
?
LAdam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/m*
_output_shapes

:PP*
dtype0
?
,Adam/simple_rnn_18/simple_rnn_cell_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*=
shared_name.,Adam/simple_rnn_18/simple_rnn_cell_18/bias/m
?
@Adam/simple_rnn_18/simple_rnn_cell_18/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_18/simple_rnn_cell_18/bias/m*
_output_shapes
:P*
dtype0
?
.Adam/simple_rnn_19/simple_rnn_cell_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*?
shared_name0.Adam/simple_rnn_19/simple_rnn_cell_19/kernel/m
?
BAdam/simple_rnn_19/simple_rnn_cell_19/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_19/simple_rnn_cell_19/kernel/m*
_output_shapes

:Pd*
dtype0
?
8Adam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*I
shared_name:8Adam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/m
?
LAdam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/m*
_output_shapes

:dd*
dtype0
?
,Adam/simple_rnn_19/simple_rnn_cell_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*=
shared_name.,Adam/simple_rnn_19/simple_rnn_cell_19/bias/m
?
@Adam/simple_rnn_19/simple_rnn_cell_19/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_19/simple_rnn_cell_19/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:*
dtype0
?
.Adam/simple_rnn_18/simple_rnn_cell_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*?
shared_name0.Adam/simple_rnn_18/simple_rnn_cell_18/kernel/v
?
BAdam/simple_rnn_18/simple_rnn_cell_18/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_18/simple_rnn_cell_18/kernel/v*
_output_shapes

:P*
dtype0
?
8Adam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*I
shared_name:8Adam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/v
?
LAdam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/v*
_output_shapes

:PP*
dtype0
?
,Adam/simple_rnn_18/simple_rnn_cell_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*=
shared_name.,Adam/simple_rnn_18/simple_rnn_cell_18/bias/v
?
@Adam/simple_rnn_18/simple_rnn_cell_18/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_18/simple_rnn_cell_18/bias/v*
_output_shapes
:P*
dtype0
?
.Adam/simple_rnn_19/simple_rnn_cell_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*?
shared_name0.Adam/simple_rnn_19/simple_rnn_cell_19/kernel/v
?
BAdam/simple_rnn_19/simple_rnn_cell_19/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_19/simple_rnn_cell_19/kernel/v*
_output_shapes

:Pd*
dtype0
?
8Adam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*I
shared_name:8Adam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/v
?
LAdam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/v*
_output_shapes

:dd*
dtype0
?
,Adam/simple_rnn_19/simple_rnn_cell_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*=
shared_name.,Adam/simple_rnn_19/simple_rnn_cell_19/bias/v
?
@Adam/simple_rnn_19/simple_rnn_cell_19/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_19/simple_rnn_cell_19/bias/v*
_output_shapes
:d*
dtype0

NoOpNoOp
?G
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
VARIABLE_VALUEdense_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ga
VARIABLE_VALUE'simple_rnn_18/simple_rnn_cell_18/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1simple_rnn_18/simple_rnn_cell_18/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_18/simple_rnn_cell_18/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'simple_rnn_19/simple_rnn_cell_19/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1simple_rnn_19/simple_rnn_cell_19/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_19/simple_rnn_cell_19/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/simple_rnn_18/simple_rnn_cell_18/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_18/simple_rnn_cell_18/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/simple_rnn_19/simple_rnn_cell_19/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_19/simple_rnn_cell_19/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/simple_rnn_18/simple_rnn_cell_18/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_18/simple_rnn_cell_18/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/simple_rnn_19/simple_rnn_cell_19/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_19/simple_rnn_cell_19/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
#serving_default_simple_rnn_18_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall#serving_default_simple_rnn_18_input'simple_rnn_18/simple_rnn_cell_18/kernel%simple_rnn_18/simple_rnn_cell_18/bias1simple_rnn_18/simple_rnn_cell_18/recurrent_kernel'simple_rnn_19/simple_rnn_cell_19/kernel%simple_rnn_19/simple_rnn_cell_19/bias1simple_rnn_19/simple_rnn_cell_19/recurrent_kerneldense_9/kerneldense_9/bias*
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
&__inference_signature_wrapper_11518256
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp;simple_rnn_18/simple_rnn_cell_18/kernel/Read/ReadVariableOpEsimple_rnn_18/simple_rnn_cell_18/recurrent_kernel/Read/ReadVariableOp9simple_rnn_18/simple_rnn_cell_18/bias/Read/ReadVariableOp;simple_rnn_19/simple_rnn_cell_19/kernel/Read/ReadVariableOpEsimple_rnn_19/simple_rnn_cell_19/recurrent_kernel/Read/ReadVariableOp9simple_rnn_19/simple_rnn_cell_19/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOpBAdam/simple_rnn_18/simple_rnn_cell_18/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_18/simple_rnn_cell_18/bias/m/Read/ReadVariableOpBAdam/simple_rnn_19/simple_rnn_cell_19/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_19/simple_rnn_cell_19/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOpBAdam/simple_rnn_18/simple_rnn_cell_18/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_18/simple_rnn_cell_18/bias/v/Read/ReadVariableOpBAdam/simple_rnn_19/simple_rnn_cell_19/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_19/simple_rnn_cell_19/bias/v/Read/ReadVariableOpConst*,
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
!__inference__traced_save_11519521
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate'simple_rnn_18/simple_rnn_cell_18/kernel1simple_rnn_18/simple_rnn_cell_18/recurrent_kernel%simple_rnn_18/simple_rnn_cell_18/bias'simple_rnn_19/simple_rnn_cell_19/kernel1simple_rnn_19/simple_rnn_cell_19/recurrent_kernel%simple_rnn_19/simple_rnn_cell_19/biastotalcountAdam/dense_9/kernel/mAdam/dense_9/bias/m.Adam/simple_rnn_18/simple_rnn_cell_18/kernel/m8Adam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/m,Adam/simple_rnn_18/simple_rnn_cell_18/bias/m.Adam/simple_rnn_19/simple_rnn_cell_19/kernel/m8Adam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/m,Adam/simple_rnn_19/simple_rnn_cell_19/bias/mAdam/dense_9/kernel/vAdam/dense_9/bias/v.Adam/simple_rnn_18/simple_rnn_cell_18/kernel/v8Adam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/v,Adam/simple_rnn_18/simple_rnn_cell_18/bias/v.Adam/simple_rnn_19/simple_rnn_cell_19/kernel/v8Adam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/v,Adam/simple_rnn_19/simple_rnn_cell_19/bias/v*+
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
$__inference__traced_restore_11519624??
?!
?
while_body_11516467
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_178_11516489_0:P2
$while_simple_rnn_cell_178_11516491_0:P6
$while_simple_rnn_cell_178_11516493_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_178_11516489:P0
"while_simple_rnn_cell_178_11516491:P4
"while_simple_rnn_cell_178_11516493:PP??1while/simple_rnn_cell_178/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/simple_rnn_cell_178/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_178_11516489_0$while_simple_rnn_cell_178_11516491_0$while_simple_rnn_cell_178_11516493_0*
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
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11516454?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_178/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_178/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp2^while/simple_rnn_cell_178/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_178_11516489$while_simple_rnn_cell_178_11516489_0"J
"while_simple_rnn_cell_178_11516491$while_simple_rnn_cell_178_11516491_0"J
"while_simple_rnn_cell_178_11516493$while_simple_rnn_cell_178_11516493_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2f
1while/simple_rnn_cell_178/StatefulPartitionedCall1while/simple_rnn_cell_178/StatefulPartitionedCall: 
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
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11518911
inputs_0D
2simple_rnn_cell_179_matmul_readvariableop_resource:PdA
3simple_rnn_cell_179_biasadd_readvariableop_resource:dF
4simple_rnn_cell_179_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_179/BiasAdd/ReadVariableOp?)simple_rnn_cell_179/MatMul/ReadVariableOp?+simple_rnn_cell_179/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_179_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_179/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_179_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_179/BiasAddBiasAdd$simple_rnn_cell_179/MatMul:product:02simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_179_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_179/MatMul_1MatMulzeros:output:03simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_179/addAddV2$simple_rnn_cell_179/BiasAdd:output:0&simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_179/TanhTanhsimple_rnn_cell_179/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_179_matmul_readvariableop_resource3simple_rnn_cell_179_biasadd_readvariableop_resource4simple_rnn_cell_179_matmul_1_readvariableop_resource*
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
while_body_11518845*
condR
while_cond_11518844*8
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
NoOpNoOp+^simple_rnn_cell_179/BiasAdd/ReadVariableOp*^simple_rnn_cell_179/MatMul/ReadVariableOp,^simple_rnn_cell_179/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2X
*simple_rnn_cell_179/BiasAdd/ReadVariableOp*simple_rnn_cell_179/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_179/MatMul/ReadVariableOp)simple_rnn_cell_179/MatMul/ReadVariableOp2Z
+simple_rnn_cell_179/MatMul_1/ReadVariableOp+simple_rnn_cell_179/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?	
?
/__inference_sequential_9_layer_call_fn_11517779

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
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517641o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
H__inference_dropout_19_layer_call_and_return_conditional_losses_11519250

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
!simple_rnn_19_while_cond_115181528
4simple_rnn_19_while_simple_rnn_19_while_loop_counter>
:simple_rnn_19_while_simple_rnn_19_while_maximum_iterations#
simple_rnn_19_while_placeholder%
!simple_rnn_19_while_placeholder_1%
!simple_rnn_19_while_placeholder_2:
6simple_rnn_19_while_less_simple_rnn_19_strided_slice_1R
Nsimple_rnn_19_while_simple_rnn_19_while_cond_11518152___redundant_placeholder0R
Nsimple_rnn_19_while_simple_rnn_19_while_cond_11518152___redundant_placeholder1R
Nsimple_rnn_19_while_simple_rnn_19_while_cond_11518152___redundant_placeholder2R
Nsimple_rnn_19_while_simple_rnn_19_while_cond_11518152___redundant_placeholder3 
simple_rnn_19_while_identity
?
simple_rnn_19/while/LessLesssimple_rnn_19_while_placeholder6simple_rnn_19_while_less_simple_rnn_19_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_19/while/IdentityIdentitysimple_rnn_19/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_19_while_identity%simple_rnn_19/while/Identity:output:0*(
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
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11516822

inputs.
simple_rnn_cell_179_11516747:Pd*
simple_rnn_cell_179_11516749:d.
simple_rnn_cell_179_11516751:dd
identity??+simple_rnn_cell_179/StatefulPartitionedCall?while;
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
+simple_rnn_cell_179/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_179_11516747simple_rnn_cell_179_11516749simple_rnn_cell_179_11516751*
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
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11516746n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_179_11516747simple_rnn_cell_179_11516749simple_rnn_cell_179_11516751*
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
while_body_11516759*
condR
while_cond_11516758*8
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
NoOpNoOp,^simple_rnn_cell_179/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2Z
+simple_rnn_cell_179/StatefulPartitionedCall+simple_rnn_cell_179/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?

?
6__inference_simple_rnn_cell_178_layer_call_fn_11519295

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
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11516454o
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
?:
?
!simple_rnn_19_while_body_115181538
4simple_rnn_19_while_simple_rnn_19_while_loop_counter>
:simple_rnn_19_while_simple_rnn_19_while_maximum_iterations#
simple_rnn_19_while_placeholder%
!simple_rnn_19_while_placeholder_1%
!simple_rnn_19_while_placeholder_27
3simple_rnn_19_while_simple_rnn_19_strided_slice_1_0s
osimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0Z
Hsimple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resource_0:PdW
Isimple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resource_0:d\
Jsimple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0:dd 
simple_rnn_19_while_identity"
simple_rnn_19_while_identity_1"
simple_rnn_19_while_identity_2"
simple_rnn_19_while_identity_3"
simple_rnn_19_while_identity_45
1simple_rnn_19_while_simple_rnn_19_strided_slice_1q
msimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensorX
Fsimple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resource:PdU
Gsimple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resource:dZ
Hsimple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resource:dd??>simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOp?=simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOp??simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp?
Esimple_rnn_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
7simple_rnn_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_19_while_placeholderNsimple_rnn_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
=simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOpHsimple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
.simple_rnn_19/while/simple_rnn_cell_179/MatMulMatMul>simple_rnn_19/while/TensorArrayV2Read/TensorListGetItem:item:0Esimple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
>simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOpIsimple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
/simple_rnn_19/while/simple_rnn_cell_179/BiasAddBiasAdd8simple_rnn_19/while/simple_rnn_cell_179/MatMul:product:0Fsimple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
?simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOpJsimple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
0simple_rnn_19/while/simple_rnn_cell_179/MatMul_1MatMul!simple_rnn_19_while_placeholder_2Gsimple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_19/while/simple_rnn_cell_179/addAddV28simple_rnn_19/while/simple_rnn_cell_179/BiasAdd:output:0:simple_rnn_19/while/simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
,simple_rnn_19/while/simple_rnn_cell_179/TanhTanh/simple_rnn_19/while/simple_rnn_cell_179/add:z:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_19_while_placeholder_1simple_rnn_19_while_placeholder0simple_rnn_19/while/simple_rnn_cell_179/Tanh:y:0*
_output_shapes
: *
element_dtype0:???[
simple_rnn_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_19/while/addAddV2simple_rnn_19_while_placeholder"simple_rnn_19/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_19/while/add_1AddV24simple_rnn_19_while_simple_rnn_19_while_loop_counter$simple_rnn_19/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_19/while/IdentityIdentitysimple_rnn_19/while/add_1:z:0^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_19/while/Identity_1Identity:simple_rnn_19_while_simple_rnn_19_while_maximum_iterations^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_19/while/Identity_2Identitysimple_rnn_19/while/add:z:0^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_19/while/Identity_3IdentityHsimple_rnn_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_19/while/Identity_4Identity0simple_rnn_19/while/simple_rnn_cell_179/Tanh:y:0^simple_rnn_19/while/NoOp*
T0*'
_output_shapes
:?????????d?
simple_rnn_19/while/NoOpNoOp?^simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOp>^simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOp@^simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_19_while_identity%simple_rnn_19/while/Identity:output:0"I
simple_rnn_19_while_identity_1'simple_rnn_19/while/Identity_1:output:0"I
simple_rnn_19_while_identity_2'simple_rnn_19/while/Identity_2:output:0"I
simple_rnn_19_while_identity_3'simple_rnn_19/while/Identity_3:output:0"I
simple_rnn_19_while_identity_4'simple_rnn_19/while/Identity_4:output:0"h
1simple_rnn_19_while_simple_rnn_19_strided_slice_13simple_rnn_19_while_simple_rnn_19_strided_slice_1_0"?
Gsimple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resourceIsimple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resource_0"?
Hsimple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resourceJsimple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0"?
Fsimple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resourceHsimple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resource_0"?
msimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensorosimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2?
>simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOp>simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOp2~
=simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOp=simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOp2?
?simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp?simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp: 
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
while_cond_11519060
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11519060___redundant_placeholder06
2while_while_cond_11519060___redundant_placeholder16
2while_while_cond_11519060___redundant_placeholder26
2while_while_cond_11519060___redundant_placeholder3
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
while_cond_11517364
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11517364___redundant_placeholder06
2while_while_cond_11517364___redundant_placeholder16
2while_while_cond_11517364___redundant_placeholder26
2while_while_cond_11517364___redundant_placeholder3
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
?!
?
while_body_11516626
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_178_11516648_0:P2
$while_simple_rnn_cell_178_11516650_0:P6
$while_simple_rnn_cell_178_11516652_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_178_11516648:P0
"while_simple_rnn_cell_178_11516650:P4
"while_simple_rnn_cell_178_11516652:PP??1while/simple_rnn_cell_178/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/simple_rnn_cell_178/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_178_11516648_0$while_simple_rnn_cell_178_11516650_0$while_simple_rnn_cell_178_11516652_0*
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
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11516574?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_178/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_178/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp2^while/simple_rnn_cell_178/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_178_11516648$while_simple_rnn_cell_178_11516648_0"J
"while_simple_rnn_cell_178_11516650$while_simple_rnn_cell_178_11516650_0"J
"while_simple_rnn_cell_178_11516652$while_simple_rnn_cell_178_11516652_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2f
1while/simple_rnn_cell_178/StatefulPartitionedCall1while/simple_rnn_cell_178/StatefulPartitionedCall: 
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
while_body_11517365
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_179_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_179_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_179_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_179_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_179/MatMul/ReadVariableOp?1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_179_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_179/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_179/BiasAddBiasAdd*while/simple_rnn_cell_179/MatMul:product:08while/simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_179/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_179/addAddV2*while/simple_rnn_cell_179/BiasAdd:output:0,while/simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_179/TanhTanh!while/simple_rnn_cell_179/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_179/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_179/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_179/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_179/MatMul/ReadVariableOp2^while/simple_rnn_cell_179/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_179_biasadd_readvariableop_resource;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_179_matmul_1_readvariableop_resource<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_179_matmul_readvariableop_resource:while_simple_rnn_cell_179_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_179/MatMul/ReadVariableOp/while/simple_rnn_cell_179/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp: 
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
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11519235

inputsD
2simple_rnn_cell_179_matmul_readvariableop_resource:PdA
3simple_rnn_cell_179_biasadd_readvariableop_resource:dF
4simple_rnn_cell_179_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_179/BiasAdd/ReadVariableOp?)simple_rnn_cell_179/MatMul/ReadVariableOp?+simple_rnn_cell_179/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_179_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_179/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_179_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_179/BiasAddBiasAdd$simple_rnn_cell_179/MatMul:product:02simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_179_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_179/MatMul_1MatMulzeros:output:03simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_179/addAddV2$simple_rnn_cell_179/BiasAdd:output:0&simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_179/TanhTanhsimple_rnn_cell_179/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_179_matmul_readvariableop_resource3simple_rnn_cell_179_biasadd_readvariableop_resource4simple_rnn_cell_179_matmul_1_readvariableop_resource*
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
while_body_11519169*
condR
while_cond_11519168*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_179/BiasAdd/ReadVariableOp*^simple_rnn_cell_179/MatMul/ReadVariableOp,^simple_rnn_cell_179/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_179/BiasAdd/ReadVariableOp*simple_rnn_cell_179/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_179/MatMul/ReadVariableOp)simple_rnn_cell_179/MatMul/ReadVariableOp2Z
+simple_rnn_cell_179/MatMul_1/ReadVariableOp+simple_rnn_cell_179/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?-
?
while_body_11517518
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_178_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_178_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_178_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_178_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_178/MatMul/ReadVariableOp?1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_178_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_178/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_178/BiasAddBiasAdd*while/simple_rnn_cell_178/MatMul:product:08while/simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_178/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_178/addAddV2*while/simple_rnn_cell_178/BiasAdd:output:0,while/simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_178/TanhTanh!while/simple_rnn_cell_178/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_178/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_178/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_178/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_178/MatMul/ReadVariableOp2^while/simple_rnn_cell_178/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_178_biasadd_readvariableop_resource;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_178_matmul_1_readvariableop_resource<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_178_matmul_readvariableop_resource:while_simple_rnn_cell_178_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_178/MatMul/ReadVariableOp/while/simple_rnn_cell_178/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp: 
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
while_body_11517160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_179_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_179_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_179_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_179_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_179/MatMul/ReadVariableOp?1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_179_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_179/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_179/BiasAddBiasAdd*while/simple_rnn_cell_179/MatMul:product:08while/simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_179/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_179/addAddV2*while/simple_rnn_cell_179/BiasAdd:output:0,while/simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_179/TanhTanh!while/simple_rnn_cell_179/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_179/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_179/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_179/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_179/MatMul/ReadVariableOp2^while/simple_rnn_cell_179/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_179_biasadd_readvariableop_resource;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_179_matmul_1_readvariableop_resource<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_179_matmul_readvariableop_resource:while_simple_rnn_cell_179_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_179/MatMul/ReadVariableOp/while/simple_rnn_cell_179/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp: 
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
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11517226

inputsD
2simple_rnn_cell_179_matmul_readvariableop_resource:PdA
3simple_rnn_cell_179_biasadd_readvariableop_resource:dF
4simple_rnn_cell_179_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_179/BiasAdd/ReadVariableOp?)simple_rnn_cell_179/MatMul/ReadVariableOp?+simple_rnn_cell_179/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_179_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_179/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_179_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_179/BiasAddBiasAdd$simple_rnn_cell_179/MatMul:product:02simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_179_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_179/MatMul_1MatMulzeros:output:03simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_179/addAddV2$simple_rnn_cell_179/BiasAdd:output:0&simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_179/TanhTanhsimple_rnn_cell_179/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_179_matmul_readvariableop_resource3simple_rnn_cell_179_biasadd_readvariableop_resource4simple_rnn_cell_179_matmul_1_readvariableop_resource*
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
while_body_11517160*
condR
while_cond_11517159*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_179/BiasAdd/ReadVariableOp*^simple_rnn_cell_179/MatMul/ReadVariableOp,^simple_rnn_cell_179/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_179/BiasAdd/ReadVariableOp*simple_rnn_cell_179/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_179/MatMul/ReadVariableOp)simple_rnn_cell_179/MatMul/ReadVariableOp2Z
+simple_rnn_cell_179/MatMul_1/ReadVariableOp+simple_rnn_cell_179/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
ق
?
$__inference__traced_restore_11519624
file_prefix1
assignvariableop_dense_9_kernel:d-
assignvariableop_1_dense_9_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: L
:assignvariableop_7_simple_rnn_18_simple_rnn_cell_18_kernel:PV
Dassignvariableop_8_simple_rnn_18_simple_rnn_cell_18_recurrent_kernel:PPF
8assignvariableop_9_simple_rnn_18_simple_rnn_cell_18_bias:PM
;assignvariableop_10_simple_rnn_19_simple_rnn_cell_19_kernel:PdW
Eassignvariableop_11_simple_rnn_19_simple_rnn_cell_19_recurrent_kernel:ddG
9assignvariableop_12_simple_rnn_19_simple_rnn_cell_19_bias:d#
assignvariableop_13_total: #
assignvariableop_14_count: ;
)assignvariableop_15_adam_dense_9_kernel_m:d5
'assignvariableop_16_adam_dense_9_bias_m:T
Bassignvariableop_17_adam_simple_rnn_18_simple_rnn_cell_18_kernel_m:P^
Lassignvariableop_18_adam_simple_rnn_18_simple_rnn_cell_18_recurrent_kernel_m:PPN
@assignvariableop_19_adam_simple_rnn_18_simple_rnn_cell_18_bias_m:PT
Bassignvariableop_20_adam_simple_rnn_19_simple_rnn_cell_19_kernel_m:Pd^
Lassignvariableop_21_adam_simple_rnn_19_simple_rnn_cell_19_recurrent_kernel_m:ddN
@assignvariableop_22_adam_simple_rnn_19_simple_rnn_cell_19_bias_m:d;
)assignvariableop_23_adam_dense_9_kernel_v:d5
'assignvariableop_24_adam_dense_9_bias_v:T
Bassignvariableop_25_adam_simple_rnn_18_simple_rnn_cell_18_kernel_v:P^
Lassignvariableop_26_adam_simple_rnn_18_simple_rnn_cell_18_recurrent_kernel_v:PPN
@assignvariableop_27_adam_simple_rnn_18_simple_rnn_cell_18_bias_v:PT
Bassignvariableop_28_adam_simple_rnn_19_simple_rnn_cell_19_kernel_v:Pd^
Lassignvariableop_29_adam_simple_rnn_19_simple_rnn_cell_19_recurrent_kernel_v:ddN
@assignvariableop_30_adam_simple_rnn_19_simple_rnn_cell_19_bias_v:d
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
AssignVariableOpAssignVariableOpassignvariableop_dense_9_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_9_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp:assignvariableop_7_simple_rnn_18_simple_rnn_cell_18_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpDassignvariableop_8_simple_rnn_18_simple_rnn_cell_18_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp8assignvariableop_9_simple_rnn_18_simple_rnn_cell_18_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp;assignvariableop_10_simple_rnn_19_simple_rnn_cell_19_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpEassignvariableop_11_simple_rnn_19_simple_rnn_cell_19_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp9assignvariableop_12_simple_rnn_19_simple_rnn_cell_19_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_9_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_9_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpBassignvariableop_17_adam_simple_rnn_18_simple_rnn_cell_18_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpLassignvariableop_18_adam_simple_rnn_18_simple_rnn_cell_18_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_simple_rnn_18_simple_rnn_cell_18_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpBassignvariableop_20_adam_simple_rnn_19_simple_rnn_cell_19_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpLassignvariableop_21_adam_simple_rnn_19_simple_rnn_cell_19_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp@assignvariableop_22_adam_simple_rnn_19_simple_rnn_cell_19_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_9_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_9_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpBassignvariableop_25_adam_simple_rnn_18_simple_rnn_cell_18_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpLassignvariableop_26_adam_simple_rnn_18_simple_rnn_cell_18_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_simple_rnn_18_simple_rnn_cell_18_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpBassignvariableop_28_adam_simple_rnn_19_simple_rnn_cell_19_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpLassignvariableop_29_adam_simple_rnn_19_simple_rnn_cell_19_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_simple_rnn_19_simple_rnn_cell_19_bias_vIdentity_30:output:0"/device:CPU:0*
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
?
?
while_cond_11518341
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11518341___redundant_placeholder06
2while_while_cond_11518341___redundant_placeholder16
2while_while_cond_11518341___redundant_placeholder26
2while_while_cond_11518341___redundant_placeholder3
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
while_body_11519169
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_179_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_179_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_179_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_179_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_179/MatMul/ReadVariableOp?1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_179_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_179/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_179/BiasAddBiasAdd*while/simple_rnn_cell_179/MatMul:product:08while/simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_179/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_179/addAddV2*while/simple_rnn_cell_179/BiasAdd:output:0,while/simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_179/TanhTanh!while/simple_rnn_cell_179/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_179/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_179/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_179/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_179/MatMul/ReadVariableOp2^while/simple_rnn_cell_179/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_179_biasadd_readvariableop_resource;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_179_matmul_1_readvariableop_resource<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_179_matmul_readvariableop_resource:while_simple_rnn_cell_179_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_179/MatMul/ReadVariableOp/while/simple_rnn_cell_179/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp: 
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
while_cond_11517159
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11517159___redundant_placeholder06
2while_while_cond_11517159___redundant_placeholder16
2while_while_cond_11517159___redundant_placeholder26
2while_while_cond_11517159___redundant_placeholder3
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
?
?
0__inference_simple_rnn_18_layer_call_fn_11518267
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11516530|
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
?
I
-__inference_dropout_18_layer_call_fn_11518737

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
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_11517117d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
f
-__inference_dropout_19_layer_call_fn_11519245

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_11517307o
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
?	
?
&__inference_signature_wrapper_11518256
simple_rnn_18_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_11516406o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_18_input
?-
?
while_body_11518666
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_178_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_178_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_178_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_178_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_178/MatMul/ReadVariableOp?1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_178_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_178/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_178/BiasAddBiasAdd*while/simple_rnn_cell_178/MatMul:product:08while/simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_178/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_178/addAddV2*while/simple_rnn_cell_178/BiasAdd:output:0,while/simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_178/TanhTanh!while/simple_rnn_cell_178/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_178/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_178/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_178/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_178/MatMul/ReadVariableOp2^while/simple_rnn_cell_178/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_178_biasadd_readvariableop_resource;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_178_matmul_1_readvariableop_resource<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_178_matmul_readvariableop_resource:while_simple_rnn_cell_178_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_178/MatMul/ReadVariableOp/while/simple_rnn_cell_178/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp: 
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
0__inference_simple_rnn_19_layer_call_fn_11518792

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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11517226o
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
:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?:
?
!simple_rnn_18_while_body_115178218
4simple_rnn_18_while_simple_rnn_18_while_loop_counter>
:simple_rnn_18_while_simple_rnn_18_while_maximum_iterations#
simple_rnn_18_while_placeholder%
!simple_rnn_18_while_placeholder_1%
!simple_rnn_18_while_placeholder_27
3simple_rnn_18_while_simple_rnn_18_strided_slice_1_0s
osimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0Z
Hsimple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resource_0:PW
Isimple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resource_0:P\
Jsimple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0:PP 
simple_rnn_18_while_identity"
simple_rnn_18_while_identity_1"
simple_rnn_18_while_identity_2"
simple_rnn_18_while_identity_3"
simple_rnn_18_while_identity_45
1simple_rnn_18_while_simple_rnn_18_strided_slice_1q
msimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensorX
Fsimple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resource:PU
Gsimple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resource:PZ
Hsimple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resource:PP??>simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOp?=simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOp??simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp?
Esimple_rnn_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
7simple_rnn_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_18_while_placeholderNsimple_rnn_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
=simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOpHsimple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
.simple_rnn_18/while/simple_rnn_cell_178/MatMulMatMul>simple_rnn_18/while/TensorArrayV2Read/TensorListGetItem:item:0Esimple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
>simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOpIsimple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
/simple_rnn_18/while/simple_rnn_cell_178/BiasAddBiasAdd8simple_rnn_18/while/simple_rnn_cell_178/MatMul:product:0Fsimple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
?simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOpJsimple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
0simple_rnn_18/while/simple_rnn_cell_178/MatMul_1MatMul!simple_rnn_18_while_placeholder_2Gsimple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_18/while/simple_rnn_cell_178/addAddV28simple_rnn_18/while/simple_rnn_cell_178/BiasAdd:output:0:simple_rnn_18/while/simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
,simple_rnn_18/while/simple_rnn_cell_178/TanhTanh/simple_rnn_18/while/simple_rnn_cell_178/add:z:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_18_while_placeholder_1simple_rnn_18_while_placeholder0simple_rnn_18/while/simple_rnn_cell_178/Tanh:y:0*
_output_shapes
: *
element_dtype0:???[
simple_rnn_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_18/while/addAddV2simple_rnn_18_while_placeholder"simple_rnn_18/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_18/while/add_1AddV24simple_rnn_18_while_simple_rnn_18_while_loop_counter$simple_rnn_18/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_18/while/IdentityIdentitysimple_rnn_18/while/add_1:z:0^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_18/while/Identity_1Identity:simple_rnn_18_while_simple_rnn_18_while_maximum_iterations^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_18/while/Identity_2Identitysimple_rnn_18/while/add:z:0^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_18/while/Identity_3IdentityHsimple_rnn_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_18/while/Identity_4Identity0simple_rnn_18/while/simple_rnn_cell_178/Tanh:y:0^simple_rnn_18/while/NoOp*
T0*'
_output_shapes
:?????????P?
simple_rnn_18/while/NoOpNoOp?^simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOp>^simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOp@^simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_18_while_identity%simple_rnn_18/while/Identity:output:0"I
simple_rnn_18_while_identity_1'simple_rnn_18/while/Identity_1:output:0"I
simple_rnn_18_while_identity_2'simple_rnn_18/while/Identity_2:output:0"I
simple_rnn_18_while_identity_3'simple_rnn_18/while/Identity_3:output:0"I
simple_rnn_18_while_identity_4'simple_rnn_18/while/Identity_4:output:0"h
1simple_rnn_18_while_simple_rnn_18_strided_slice_13simple_rnn_18_while_simple_rnn_18_strided_slice_1_0"?
Gsimple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resourceIsimple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resource_0"?
Hsimple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resourceJsimple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0"?
Fsimple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resourceHsimple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resource_0"?
msimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensorosimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2?
>simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOp>simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOp2~
=simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOp=simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOp2?
?simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp?simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp: 
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
g
H__inference_dropout_19_layer_call_and_return_conditional_losses_11519262

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
?
?
while_cond_11517517
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11517517___redundant_placeholder06
2while_while_cond_11517517___redundant_placeholder16
2while_while_cond_11517517___redundant_placeholder26
2while_while_cond_11517517___redundant_placeholder3
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
g
H__inference_dropout_19_layer_call_and_return_conditional_losses_11517307

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
?	
?
E__inference_dense_9_layer_call_and_return_conditional_losses_11519281

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
??
?	
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517999

inputsR
@simple_rnn_18_simple_rnn_cell_178_matmul_readvariableop_resource:PO
Asimple_rnn_18_simple_rnn_cell_178_biasadd_readvariableop_resource:PT
Bsimple_rnn_18_simple_rnn_cell_178_matmul_1_readvariableop_resource:PPR
@simple_rnn_19_simple_rnn_cell_179_matmul_readvariableop_resource:PdO
Asimple_rnn_19_simple_rnn_cell_179_biasadd_readvariableop_resource:dT
Bsimple_rnn_19_simple_rnn_cell_179_matmul_1_readvariableop_resource:dd8
&dense_9_matmul_readvariableop_resource:d5
'dense_9_biasadd_readvariableop_resource:
identity??dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?8simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOp?7simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOp?9simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp?simple_rnn_18/while?8simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOp?7simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOp?9simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp?simple_rnn_19/whileI
simple_rnn_18/ShapeShapeinputs*
T0*
_output_shapes
:k
!simple_rnn_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_18/strided_sliceStridedSlicesimple_rnn_18/Shape:output:0*simple_rnn_18/strided_slice/stack:output:0,simple_rnn_18/strided_slice/stack_1:output:0,simple_rnn_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
simple_rnn_18/zeros/packedPack$simple_rnn_18/strided_slice:output:0%simple_rnn_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_18/zerosFill#simple_rnn_18/zeros/packed:output:0"simple_rnn_18/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pq
simple_rnn_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_18/transpose	Transposeinputs%simple_rnn_18/transpose/perm:output:0*
T0*+
_output_shapes
:?????????`
simple_rnn_18/Shape_1Shapesimple_rnn_18/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_18/strided_slice_1StridedSlicesimple_rnn_18/Shape_1:output:0,simple_rnn_18/strided_slice_1/stack:output:0.simple_rnn_18/strided_slice_1/stack_1:output:0.simple_rnn_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_18/TensorArrayV2TensorListReserve2simple_rnn_18/TensorArrayV2/element_shape:output:0&simple_rnn_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Csimple_rnn_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
5simple_rnn_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_18/transpose:y:0Lsimple_rnn_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???m
#simple_rnn_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_18/strided_slice_2StridedSlicesimple_rnn_18/transpose:y:0,simple_rnn_18/strided_slice_2/stack:output:0.simple_rnn_18/strided_slice_2/stack_1:output:0.simple_rnn_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
7simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp@simple_rnn_18_simple_rnn_cell_178_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
(simple_rnn_18/simple_rnn_cell_178/MatMulMatMul&simple_rnn_18/strided_slice_2:output:0?simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOpAsimple_rnn_18_simple_rnn_cell_178_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
)simple_rnn_18/simple_rnn_cell_178/BiasAddBiasAdd2simple_rnn_18/simple_rnn_cell_178/MatMul:product:0@simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
9simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOpBsimple_rnn_18_simple_rnn_cell_178_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
*simple_rnn_18/simple_rnn_cell_178/MatMul_1MatMulsimple_rnn_18/zeros:output:0Asimple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
%simple_rnn_18/simple_rnn_cell_178/addAddV22simple_rnn_18/simple_rnn_cell_178/BiasAdd:output:04simple_rnn_18/simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
&simple_rnn_18/simple_rnn_cell_178/TanhTanh)simple_rnn_18/simple_rnn_cell_178/add:z:0*
T0*'
_output_shapes
:?????????P|
+simple_rnn_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
simple_rnn_18/TensorArrayV2_1TensorListReserve4simple_rnn_18/TensorArrayV2_1/element_shape:output:0&simple_rnn_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???T
simple_rnn_18/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 simple_rnn_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_18/whileWhile)simple_rnn_18/while/loop_counter:output:0/simple_rnn_18/while/maximum_iterations:output:0simple_rnn_18/time:output:0&simple_rnn_18/TensorArrayV2_1:handle:0simple_rnn_18/zeros:output:0&simple_rnn_18/strided_slice_1:output:0Esimple_rnn_18/TensorArrayUnstack/TensorListFromTensor:output_handle:0@simple_rnn_18_simple_rnn_cell_178_matmul_readvariableop_resourceAsimple_rnn_18_simple_rnn_cell_178_biasadd_readvariableop_resourceBsimple_rnn_18_simple_rnn_cell_178_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *-
body%R#
!simple_rnn_18_while_body_11517821*-
cond%R#
!simple_rnn_18_while_cond_11517820*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
>simple_rnn_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
0simple_rnn_18/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_18/while:output:3Gsimple_rnn_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0v
#simple_rnn_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%simple_rnn_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_18/strided_slice_3StridedSlice9simple_rnn_18/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_18/strided_slice_3/stack:output:0.simple_rnn_18/strided_slice_3/stack_1:output:0.simple_rnn_18/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_masks
simple_rnn_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_18/transpose_1	Transpose9simple_rnn_18/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_18/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pt
dropout_18/IdentityIdentitysimple_rnn_18/transpose_1:y:0*
T0*+
_output_shapes
:?????????P_
simple_rnn_19/ShapeShapedropout_18/Identity:output:0*
T0*
_output_shapes
:k
!simple_rnn_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_19/strided_sliceStridedSlicesimple_rnn_19/Shape:output:0*simple_rnn_19/strided_slice/stack:output:0,simple_rnn_19/strided_slice/stack_1:output:0,simple_rnn_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
simple_rnn_19/zeros/packedPack$simple_rnn_19/strided_slice:output:0%simple_rnn_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_19/zerosFill#simple_rnn_19/zeros/packed:output:0"simple_rnn_19/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dq
simple_rnn_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_19/transpose	Transposedropout_18/Identity:output:0%simple_rnn_19/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P`
simple_rnn_19/Shape_1Shapesimple_rnn_19/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_19/strided_slice_1StridedSlicesimple_rnn_19/Shape_1:output:0,simple_rnn_19/strided_slice_1/stack:output:0.simple_rnn_19/strided_slice_1/stack_1:output:0.simple_rnn_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_19/TensorArrayV2TensorListReserve2simple_rnn_19/TensorArrayV2/element_shape:output:0&simple_rnn_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Csimple_rnn_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
5simple_rnn_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_19/transpose:y:0Lsimple_rnn_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???m
#simple_rnn_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_19/strided_slice_2StridedSlicesimple_rnn_19/transpose:y:0,simple_rnn_19/strided_slice_2/stack:output:0.simple_rnn_19/strided_slice_2/stack_1:output:0.simple_rnn_19/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
7simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp@simple_rnn_19_simple_rnn_cell_179_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
(simple_rnn_19/simple_rnn_cell_179/MatMulMatMul&simple_rnn_19/strided_slice_2:output:0?simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOpAsimple_rnn_19_simple_rnn_cell_179_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
)simple_rnn_19/simple_rnn_cell_179/BiasAddBiasAdd2simple_rnn_19/simple_rnn_cell_179/MatMul:product:0@simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
9simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOpBsimple_rnn_19_simple_rnn_cell_179_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
*simple_rnn_19/simple_rnn_cell_179/MatMul_1MatMulsimple_rnn_19/zeros:output:0Asimple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
%simple_rnn_19/simple_rnn_cell_179/addAddV22simple_rnn_19/simple_rnn_cell_179/BiasAdd:output:04simple_rnn_19/simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
&simple_rnn_19/simple_rnn_cell_179/TanhTanh)simple_rnn_19/simple_rnn_cell_179/add:z:0*
T0*'
_output_shapes
:?????????d|
+simple_rnn_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
simple_rnn_19/TensorArrayV2_1TensorListReserve4simple_rnn_19/TensorArrayV2_1/element_shape:output:0&simple_rnn_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???T
simple_rnn_19/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 simple_rnn_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_19/whileWhile)simple_rnn_19/while/loop_counter:output:0/simple_rnn_19/while/maximum_iterations:output:0simple_rnn_19/time:output:0&simple_rnn_19/TensorArrayV2_1:handle:0simple_rnn_19/zeros:output:0&simple_rnn_19/strided_slice_1:output:0Esimple_rnn_19/TensorArrayUnstack/TensorListFromTensor:output_handle:0@simple_rnn_19_simple_rnn_cell_179_matmul_readvariableop_resourceAsimple_rnn_19_simple_rnn_cell_179_biasadd_readvariableop_resourceBsimple_rnn_19_simple_rnn_cell_179_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *-
body%R#
!simple_rnn_19_while_body_11517926*-
cond%R#
!simple_rnn_19_while_cond_11517925*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
>simple_rnn_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
0simple_rnn_19/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_19/while:output:3Gsimple_rnn_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0v
#simple_rnn_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%simple_rnn_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_19/strided_slice_3StridedSlice9simple_rnn_19/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_19/strided_slice_3/stack:output:0.simple_rnn_19/strided_slice_3/stack_1:output:0.simple_rnn_19/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_masks
simple_rnn_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_19/transpose_1	Transpose9simple_rnn_19/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_19/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dy
dropout_19/IdentityIdentity&simple_rnn_19/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_9/MatMulMatMuldropout_19/Identity:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp9^simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOp8^simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOp:^simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp^simple_rnn_18/while9^simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOp8^simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOp:^simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp^simple_rnn_19/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2t
8simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOp8simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOp2r
7simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOp7simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOp2v
9simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp9simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp2*
simple_rnn_18/whilesimple_rnn_18/while2t
8simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOp8simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOp2r
7simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOp7simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOp2v
9simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp9simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp2*
simple_rnn_19/whilesimple_rnn_19/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
!simple_rnn_18_while_cond_115180408
4simple_rnn_18_while_simple_rnn_18_while_loop_counter>
:simple_rnn_18_while_simple_rnn_18_while_maximum_iterations#
simple_rnn_18_while_placeholder%
!simple_rnn_18_while_placeholder_1%
!simple_rnn_18_while_placeholder_2:
6simple_rnn_18_while_less_simple_rnn_18_strided_slice_1R
Nsimple_rnn_18_while_simple_rnn_18_while_cond_11518040___redundant_placeholder0R
Nsimple_rnn_18_while_simple_rnn_18_while_cond_11518040___redundant_placeholder1R
Nsimple_rnn_18_while_simple_rnn_18_while_cond_11518040___redundant_placeholder2R
Nsimple_rnn_18_while_simple_rnn_18_while_cond_11518040___redundant_placeholder3 
simple_rnn_18_while_identity
?
simple_rnn_18/while/LessLesssimple_rnn_18_while_placeholder6simple_rnn_18_while_less_simple_rnn_18_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_18/while/IdentityIdentitysimple_rnn_18/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_18_while_identity%simple_rnn_18/while/Identity:output:0*(
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
?=
?
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11517584

inputsD
2simple_rnn_cell_178_matmul_readvariableop_resource:PA
3simple_rnn_cell_178_biasadd_readvariableop_resource:PF
4simple_rnn_cell_178_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_178/BiasAdd/ReadVariableOp?)simple_rnn_cell_178/MatMul/ReadVariableOp?+simple_rnn_cell_178/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_178_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_178/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_178_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_178/BiasAddBiasAdd$simple_rnn_cell_178/MatMul:product:02simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_178_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_178/MatMul_1MatMulzeros:output:03simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_178/addAddV2$simple_rnn_cell_178/BiasAdd:output:0&simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_178/TanhTanhsimple_rnn_cell_178/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_178_matmul_readvariableop_resource3simple_rnn_cell_178_biasadd_readvariableop_resource4simple_rnn_cell_178_matmul_1_readvariableop_resource*
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
while_body_11517518*
condR
while_cond_11517517*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_178/BiasAdd/ReadVariableOp*^simple_rnn_cell_178/MatMul/ReadVariableOp,^simple_rnn_cell_178/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_178/BiasAdd/ReadVariableOp*simple_rnn_cell_178/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_178/MatMul/ReadVariableOp)simple_rnn_cell_178/MatMul/ReadVariableOp2Z
+simple_rnn_cell_178/MatMul_1/ReadVariableOp+simple_rnn_cell_178/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_simple_rnn_19_layer_call_fn_11518770
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11516822o
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
?
?
0__inference_simple_rnn_19_layer_call_fn_11518781
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11516981o
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
?F
?
.sequential_9_simple_rnn_18_while_body_11516228R
Nsequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_while_loop_counterX
Tsequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_while_maximum_iterations0
,sequential_9_simple_rnn_18_while_placeholder2
.sequential_9_simple_rnn_18_while_placeholder_12
.sequential_9_simple_rnn_18_while_placeholder_2Q
Msequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_strided_slice_1_0?
?sequential_9_simple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0g
Usequential_9_simple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resource_0:Pd
Vsequential_9_simple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resource_0:Pi
Wsequential_9_simple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0:PP-
)sequential_9_simple_rnn_18_while_identity/
+sequential_9_simple_rnn_18_while_identity_1/
+sequential_9_simple_rnn_18_while_identity_2/
+sequential_9_simple_rnn_18_while_identity_3/
+sequential_9_simple_rnn_18_while_identity_4O
Ksequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_strided_slice_1?
?sequential_9_simple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_simple_rnn_18_tensorarrayunstack_tensorlistfromtensore
Ssequential_9_simple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resource:Pb
Tsequential_9_simple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resource:Pg
Usequential_9_simple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resource:PP??Ksequential_9/simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOp?Jsequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOp?Lsequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp?
Rsequential_9/simple_rnn_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Dsequential_9/simple_rnn_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_9_simple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0,sequential_9_simple_rnn_18_while_placeholder[sequential_9/simple_rnn_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
Jsequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOpUsequential_9_simple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
;sequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMulMatMulKsequential_9/simple_rnn_18/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Ksequential_9/simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOpVsequential_9_simple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
<sequential_9/simple_rnn_18/while/simple_rnn_cell_178/BiasAddBiasAddEsequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul:product:0Ssequential_9/simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Lsequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOpWsequential_9_simple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
=sequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul_1MatMul.sequential_9_simple_rnn_18_while_placeholder_2Tsequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8sequential_9/simple_rnn_18/while/simple_rnn_cell_178/addAddV2Esequential_9/simple_rnn_18/while/simple_rnn_cell_178/BiasAdd:output:0Gsequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
9sequential_9/simple_rnn_18/while/simple_rnn_cell_178/TanhTanh<sequential_9/simple_rnn_18/while/simple_rnn_cell_178/add:z:0*
T0*'
_output_shapes
:?????????P?
Esequential_9/simple_rnn_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_9_simple_rnn_18_while_placeholder_1,sequential_9_simple_rnn_18_while_placeholder=sequential_9/simple_rnn_18/while/simple_rnn_cell_178/Tanh:y:0*
_output_shapes
: *
element_dtype0:???h
&sequential_9/simple_rnn_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
$sequential_9/simple_rnn_18/while/addAddV2,sequential_9_simple_rnn_18_while_placeholder/sequential_9/simple_rnn_18/while/add/y:output:0*
T0*
_output_shapes
: j
(sequential_9/simple_rnn_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
&sequential_9/simple_rnn_18/while/add_1AddV2Nsequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_while_loop_counter1sequential_9/simple_rnn_18/while/add_1/y:output:0*
T0*
_output_shapes
: ?
)sequential_9/simple_rnn_18/while/IdentityIdentity*sequential_9/simple_rnn_18/while/add_1:z:0&^sequential_9/simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ?
+sequential_9/simple_rnn_18/while/Identity_1IdentityTsequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_while_maximum_iterations&^sequential_9/simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ?
+sequential_9/simple_rnn_18/while/Identity_2Identity(sequential_9/simple_rnn_18/while/add:z:0&^sequential_9/simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ?
+sequential_9/simple_rnn_18/while/Identity_3IdentityUsequential_9/simple_rnn_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_9/simple_rnn_18/while/NoOp*
T0*
_output_shapes
: :????
+sequential_9/simple_rnn_18/while/Identity_4Identity=sequential_9/simple_rnn_18/while/simple_rnn_cell_178/Tanh:y:0&^sequential_9/simple_rnn_18/while/NoOp*
T0*'
_output_shapes
:?????????P?
%sequential_9/simple_rnn_18/while/NoOpNoOpL^sequential_9/simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOpK^sequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOpM^sequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_9_simple_rnn_18_while_identity2sequential_9/simple_rnn_18/while/Identity:output:0"c
+sequential_9_simple_rnn_18_while_identity_14sequential_9/simple_rnn_18/while/Identity_1:output:0"c
+sequential_9_simple_rnn_18_while_identity_24sequential_9/simple_rnn_18/while/Identity_2:output:0"c
+sequential_9_simple_rnn_18_while_identity_34sequential_9/simple_rnn_18/while/Identity_3:output:0"c
+sequential_9_simple_rnn_18_while_identity_44sequential_9/simple_rnn_18/while/Identity_4:output:0"?
Ksequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_strided_slice_1Msequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_strided_slice_1_0"?
Tsequential_9_simple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resourceVsequential_9_simple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resource_0"?
Usequential_9_simple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resourceWsequential_9_simple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0"?
Ssequential_9_simple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resourceUsequential_9_simple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resource_0"?
?sequential_9_simple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor?sequential_9_simple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2?
Ksequential_9/simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOpKsequential_9/simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOp2?
Jsequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOpJsequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOp2?
Lsequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOpLsequential_9/simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp: 
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
while_body_11518450
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_178_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_178_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_178_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_178_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_178/MatMul/ReadVariableOp?1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_178_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_178/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_178/BiasAddBiasAdd*while/simple_rnn_cell_178/MatMul:product:08while/simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_178/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_178/addAddV2*while/simple_rnn_cell_178/BiasAdd:output:0,while/simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_178/TanhTanh!while/simple_rnn_cell_178/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_178/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_178/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_178/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_178/MatMul/ReadVariableOp2^while/simple_rnn_cell_178/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_178_biasadd_readvariableop_resource;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_178_matmul_1_readvariableop_resource<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_178_matmul_readvariableop_resource:while_simple_rnn_cell_178_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_178/MatMul/ReadVariableOp/while/simple_rnn_cell_178/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp: 
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
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11518516
inputs_0D
2simple_rnn_cell_178_matmul_readvariableop_resource:PA
3simple_rnn_cell_178_biasadd_readvariableop_resource:PF
4simple_rnn_cell_178_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_178/BiasAdd/ReadVariableOp?)simple_rnn_cell_178/MatMul/ReadVariableOp?+simple_rnn_cell_178/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_178_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_178/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_178_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_178/BiasAddBiasAdd$simple_rnn_cell_178/MatMul:product:02simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_178_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_178/MatMul_1MatMulzeros:output:03simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_178/addAddV2$simple_rnn_cell_178/BiasAdd:output:0&simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_178/TanhTanhsimple_rnn_cell_178/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_178_matmul_readvariableop_resource3simple_rnn_cell_178_biasadd_readvariableop_resource4simple_rnn_cell_178_matmul_1_readvariableop_resource*
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
while_body_11518450*
condR
while_cond_11518449*8
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
NoOpNoOp+^simple_rnn_cell_178/BiasAdd/ReadVariableOp*^simple_rnn_cell_178/MatMul/ReadVariableOp,^simple_rnn_cell_178/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2X
*simple_rnn_cell_178/BiasAdd/ReadVariableOp*simple_rnn_cell_178/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_178/MatMul/ReadVariableOp)simple_rnn_cell_178/MatMul/ReadVariableOp2Z
+simple_rnn_cell_178/MatMul_1/ReadVariableOp+simple_rnn_cell_178/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
f
H__inference_dropout_19_layer_call_and_return_conditional_losses_11517239

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
?
?
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11516866

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
.sequential_9_simple_rnn_19_while_cond_11516332R
Nsequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_while_loop_counterX
Tsequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_while_maximum_iterations0
,sequential_9_simple_rnn_19_while_placeholder2
.sequential_9_simple_rnn_19_while_placeholder_12
.sequential_9_simple_rnn_19_while_placeholder_2T
Psequential_9_simple_rnn_19_while_less_sequential_9_simple_rnn_19_strided_slice_1l
hsequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_while_cond_11516332___redundant_placeholder0l
hsequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_while_cond_11516332___redundant_placeholder1l
hsequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_while_cond_11516332___redundant_placeholder2l
hsequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_while_cond_11516332___redundant_placeholder3-
)sequential_9_simple_rnn_19_while_identity
?
%sequential_9/simple_rnn_19/while/LessLess,sequential_9_simple_rnn_19_while_placeholderPsequential_9_simple_rnn_19_while_less_sequential_9_simple_rnn_19_strided_slice_1*
T0*
_output_shapes
: ?
)sequential_9/simple_rnn_19/while/IdentityIdentity)sequential_9/simple_rnn_19/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_9_simple_rnn_19_while_identity2sequential_9/simple_rnn_19/while/Identity:output:0*(
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
while_body_11518845
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_179_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_179_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_179_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_179_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_179/MatMul/ReadVariableOp?1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_179_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_179/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_179/BiasAddBiasAdd*while/simple_rnn_cell_179/MatMul:product:08while/simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_179/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_179/addAddV2*while/simple_rnn_cell_179/BiasAdd:output:0,while/simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_179/TanhTanh!while/simple_rnn_cell_179/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_179/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_179/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_179/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_179/MatMul/ReadVariableOp2^while/simple_rnn_cell_179/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_179_biasadd_readvariableop_resource;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_179_matmul_1_readvariableop_resource<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_179_matmul_readvariableop_resource:while_simple_rnn_cell_179_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_179/MatMul/ReadVariableOp/while/simple_rnn_cell_179/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp: 
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
while_cond_11516917
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11516917___redundant_placeholder06
2while_while_cond_11516917___redundant_placeholder16
2while_while_cond_11516917___redundant_placeholder26
2while_while_cond_11516917___redundant_placeholder3
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
?
E__inference_dense_9_layer_call_and_return_conditional_losses_11517251

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
?4
?
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11516689

inputs.
simple_rnn_cell_178_11516614:P*
simple_rnn_cell_178_11516616:P.
simple_rnn_cell_178_11516618:PP
identity??+simple_rnn_cell_178/StatefulPartitionedCall?while;
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
+simple_rnn_cell_178/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_178_11516614simple_rnn_cell_178_11516616simple_rnn_cell_178_11516618*
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
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11516574n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_178_11516614simple_rnn_cell_178_11516616simple_rnn_cell_178_11516618*
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
while_body_11516626*
condR
while_cond_11516625*8
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
NoOpNoOp,^simple_rnn_cell_178/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+simple_rnn_cell_178/StatefulPartitionedCall+simple_rnn_cell_178/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?-
?
while_body_11518342
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_178_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_178_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_178_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_178_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_178/MatMul/ReadVariableOp?1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_178_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_178/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_178/BiasAddBiasAdd*while/simple_rnn_cell_178/MatMul:product:08while/simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_178/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_178/addAddV2*while/simple_rnn_cell_178/BiasAdd:output:0,while/simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_178/TanhTanh!while/simple_rnn_cell_178/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_178/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_178/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_178/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_178/MatMul/ReadVariableOp2^while/simple_rnn_cell_178/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_178_biasadd_readvariableop_resource;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_178_matmul_1_readvariableop_resource<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_178_matmul_readvariableop_resource:while_simple_rnn_cell_178_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_178/MatMul/ReadVariableOp/while/simple_rnn_cell_178/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp: 
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
while_cond_11519168
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11519168___redundant_placeholder06
2while_while_cond_11519168___redundant_placeholder16
2while_while_cond_11519168___redundant_placeholder26
2while_while_cond_11519168___redundant_placeholder3
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

?
6__inference_simple_rnn_cell_179_layer_call_fn_11519357

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
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11516746o
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
?
?
0__inference_simple_rnn_18_layer_call_fn_11518278
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11516689|
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
while_body_11519061
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_179_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_179_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_179_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_179_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_179/MatMul/ReadVariableOp?1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_179_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_179/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_179/BiasAddBiasAdd*while/simple_rnn_cell_179/MatMul:product:08while/simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_179/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_179/addAddV2*while/simple_rnn_cell_179/BiasAdd:output:0,while/simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_179/TanhTanh!while/simple_rnn_cell_179/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_179/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_179/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_179/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_179/MatMul/ReadVariableOp2^while/simple_rnn_cell_179/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_179_biasadd_readvariableop_resource;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_179_matmul_1_readvariableop_resource<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_179_matmul_readvariableop_resource:while_simple_rnn_cell_179_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_179/MatMul/ReadVariableOp/while/simple_rnn_cell_179/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp: 
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
?
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517731
simple_rnn_18_input(
simple_rnn_18_11517709:P$
simple_rnn_18_11517711:P(
simple_rnn_18_11517713:PP(
simple_rnn_19_11517717:Pd$
simple_rnn_19_11517719:d(
simple_rnn_19_11517721:dd"
dense_9_11517725:d
dense_9_11517727:
identity??dense_9/StatefulPartitionedCall?"dropout_18/StatefulPartitionedCall?"dropout_19/StatefulPartitionedCall?%simple_rnn_18/StatefulPartitionedCall?%simple_rnn_19/StatefulPartitionedCall?
%simple_rnn_18/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_18_inputsimple_rnn_18_11517709simple_rnn_18_11517711simple_rnn_18_11517713*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11517584?
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_11517460?
%simple_rnn_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0simple_rnn_19_11517717simple_rnn_19_11517719simple_rnn_19_11517721*
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11517431?
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_19/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_11517307?
dense_9/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_9_11517725dense_9_11517727*
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
E__inference_dense_9_layer_call_and_return_conditional_losses_11517251w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_9/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall&^simple_rnn_18/StatefulPartitionedCall&^simple_rnn_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2N
%simple_rnn_18/StatefulPartitionedCall%simple_rnn_18/StatefulPartitionedCall2N
%simple_rnn_19/StatefulPartitionedCall%simple_rnn_19/StatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_18_input
?>
?
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11519019
inputs_0D
2simple_rnn_cell_179_matmul_readvariableop_resource:PdA
3simple_rnn_cell_179_biasadd_readvariableop_resource:dF
4simple_rnn_cell_179_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_179/BiasAdd/ReadVariableOp?)simple_rnn_cell_179/MatMul/ReadVariableOp?+simple_rnn_cell_179/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_179_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_179/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_179_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_179/BiasAddBiasAdd$simple_rnn_cell_179/MatMul:product:02simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_179_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_179/MatMul_1MatMulzeros:output:03simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_179/addAddV2$simple_rnn_cell_179/BiasAdd:output:0&simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_179/TanhTanhsimple_rnn_cell_179/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_179_matmul_readvariableop_resource3simple_rnn_cell_179_biasadd_readvariableop_resource4simple_rnn_cell_179_matmul_1_readvariableop_resource*
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
while_body_11518953*
condR
while_cond_11518952*8
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
NoOpNoOp+^simple_rnn_cell_179/BiasAdd/ReadVariableOp*^simple_rnn_cell_179/MatMul/ReadVariableOp,^simple_rnn_cell_179/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2X
*simple_rnn_cell_179/BiasAdd/ReadVariableOp*simple_rnn_cell_179/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_179/MatMul/ReadVariableOp)simple_rnn_cell_179/MatMul/ReadVariableOp2Z
+simple_rnn_cell_179/MatMul_1/ReadVariableOp+simple_rnn_cell_179/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?

?
!simple_rnn_19_while_cond_115179258
4simple_rnn_19_while_simple_rnn_19_while_loop_counter>
:simple_rnn_19_while_simple_rnn_19_while_maximum_iterations#
simple_rnn_19_while_placeholder%
!simple_rnn_19_while_placeholder_1%
!simple_rnn_19_while_placeholder_2:
6simple_rnn_19_while_less_simple_rnn_19_strided_slice_1R
Nsimple_rnn_19_while_simple_rnn_19_while_cond_11517925___redundant_placeholder0R
Nsimple_rnn_19_while_simple_rnn_19_while_cond_11517925___redundant_placeholder1R
Nsimple_rnn_19_while_simple_rnn_19_while_cond_11517925___redundant_placeholder2R
Nsimple_rnn_19_while_simple_rnn_19_while_cond_11517925___redundant_placeholder3 
simple_rnn_19_while_identity
?
simple_rnn_19/while/LessLesssimple_rnn_19_while_placeholder6simple_rnn_19_while_less_simple_rnn_19_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_19/while/IdentityIdentitysimple_rnn_19/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_19_while_identity%simple_rnn_19/while/Identity:output:0*(
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
?!
?
while_body_11516918
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_179_11516940_0:Pd2
$while_simple_rnn_cell_179_11516942_0:d6
$while_simple_rnn_cell_179_11516944_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_179_11516940:Pd0
"while_simple_rnn_cell_179_11516942:d4
"while_simple_rnn_cell_179_11516944:dd??1while/simple_rnn_cell_179/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
1while/simple_rnn_cell_179/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_179_11516940_0$while_simple_rnn_cell_179_11516942_0$while_simple_rnn_cell_179_11516944_0*
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
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11516866?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_179/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_179/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp2^while/simple_rnn_cell_179/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_179_11516940$while_simple_rnn_cell_179_11516940_0"J
"while_simple_rnn_cell_179_11516942$while_simple_rnn_cell_179_11516942_0"J
"while_simple_rnn_cell_179_11516944$while_simple_rnn_cell_179_11516944_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2f
1while/simple_rnn_cell_179/StatefulPartitionedCall1while/simple_rnn_cell_179/StatefulPartitionedCall: 
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
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11519326

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
?
?
while_cond_11518844
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11518844___redundant_placeholder06
2while_while_cond_11518844___redundant_placeholder16
2while_while_cond_11518844___redundant_placeholder26
2while_while_cond_11518844___redundant_placeholder3
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
while_body_11518558
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_178_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_178_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_178_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_178_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_178/MatMul/ReadVariableOp?1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_178_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_178/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_178/BiasAddBiasAdd*while/simple_rnn_cell_178/MatMul:product:08while/simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_178/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_178/addAddV2*while/simple_rnn_cell_178/BiasAdd:output:0,while/simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_178/TanhTanh!while/simple_rnn_cell_178/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_178/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_178/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_178/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_178/MatMul/ReadVariableOp2^while/simple_rnn_cell_178/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_178_biasadd_readvariableop_resource;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_178_matmul_1_readvariableop_resource<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_178_matmul_readvariableop_resource:while_simple_rnn_cell_178_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_178/MatMul/ReadVariableOp/while/simple_rnn_cell_178/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp: 
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
while_cond_11518952
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11518952___redundant_placeholder06
2while_while_cond_11518952___redundant_placeholder16
2while_while_cond_11518952___redundant_placeholder26
2while_while_cond_11518952___redundant_placeholder3
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
while_cond_11516466
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11516466___redundant_placeholder06
2while_while_cond_11516466___redundant_placeholder16
2while_while_cond_11516466___redundant_placeholder26
2while_while_cond_11516466___redundant_placeholder3
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
?
/__inference_sequential_9_layer_call_fn_11517277
simple_rnn_18_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517258o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_18_input
?
f
-__inference_dropout_18_layer_call_fn_11518742

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
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_11517460s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
f
H__inference_dropout_18_layer_call_and_return_conditional_losses_11517117

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11516746

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
?=
?
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11518732

inputsD
2simple_rnn_cell_178_matmul_readvariableop_resource:PA
3simple_rnn_cell_178_biasadd_readvariableop_resource:PF
4simple_rnn_cell_178_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_178/BiasAdd/ReadVariableOp?)simple_rnn_cell_178/MatMul/ReadVariableOp?+simple_rnn_cell_178/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_178_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_178/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_178_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_178/BiasAddBiasAdd$simple_rnn_cell_178/MatMul:product:02simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_178_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_178/MatMul_1MatMulzeros:output:03simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_178/addAddV2$simple_rnn_cell_178/BiasAdd:output:0&simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_178/TanhTanhsimple_rnn_cell_178/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_178_matmul_readvariableop_resource3simple_rnn_cell_178_biasadd_readvariableop_resource4simple_rnn_cell_178_matmul_1_readvariableop_resource*
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
while_body_11518666*
condR
while_cond_11518665*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_178/BiasAdd/ReadVariableOp*^simple_rnn_cell_178/MatMul/ReadVariableOp,^simple_rnn_cell_178/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_178/BiasAdd/ReadVariableOp*simple_rnn_cell_178/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_178/MatMul/ReadVariableOp)simple_rnn_cell_178/MatMul/ReadVariableOp2Z
+simple_rnn_cell_178/MatMul_1/ReadVariableOp+simple_rnn_cell_178/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
6__inference_simple_rnn_cell_178_layer_call_fn_11519309

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
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11516574o
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
?
?
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11519388

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
while_body_11517038
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_178_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_178_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_178_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_178_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_178/MatMul/ReadVariableOp?1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_178_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_178/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_178/BiasAddBiasAdd*while/simple_rnn_cell_178/MatMul:product:08while/simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_178/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_178/addAddV2*while/simple_rnn_cell_178/BiasAdd:output:0,while/simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_178/TanhTanh!while/simple_rnn_cell_178/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_178/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_178/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_178/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_178/MatMul/ReadVariableOp2^while/simple_rnn_cell_178/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_178_biasadd_readvariableop_resource;while_simple_rnn_cell_178_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_178_matmul_1_readvariableop_resource<while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_178_matmul_readvariableop_resource:while_simple_rnn_cell_178_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp0while/simple_rnn_cell_178/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_178/MatMul/ReadVariableOp/while/simple_rnn_cell_178/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp1while/simple_rnn_cell_178/MatMul_1/ReadVariableOp: 
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
0__inference_simple_rnn_19_layer_call_fn_11518803

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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11517431o
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
:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
ѷ
?

#__inference__wrapped_model_11516406
simple_rnn_18_input_
Msequential_9_simple_rnn_18_simple_rnn_cell_178_matmul_readvariableop_resource:P\
Nsequential_9_simple_rnn_18_simple_rnn_cell_178_biasadd_readvariableop_resource:Pa
Osequential_9_simple_rnn_18_simple_rnn_cell_178_matmul_1_readvariableop_resource:PP_
Msequential_9_simple_rnn_19_simple_rnn_cell_179_matmul_readvariableop_resource:Pd\
Nsequential_9_simple_rnn_19_simple_rnn_cell_179_biasadd_readvariableop_resource:da
Osequential_9_simple_rnn_19_simple_rnn_cell_179_matmul_1_readvariableop_resource:ddE
3sequential_9_dense_9_matmul_readvariableop_resource:dB
4sequential_9_dense_9_biasadd_readvariableop_resource:
identity??+sequential_9/dense_9/BiasAdd/ReadVariableOp?*sequential_9/dense_9/MatMul/ReadVariableOp?Esequential_9/simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOp?Dsequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOp?Fsequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp? sequential_9/simple_rnn_18/while?Esequential_9/simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOp?Dsequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOp?Fsequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp? sequential_9/simple_rnn_19/whilec
 sequential_9/simple_rnn_18/ShapeShapesimple_rnn_18_input*
T0*
_output_shapes
:x
.sequential_9/simple_rnn_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_9/simple_rnn_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_9/simple_rnn_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential_9/simple_rnn_18/strided_sliceStridedSlice)sequential_9/simple_rnn_18/Shape:output:07sequential_9/simple_rnn_18/strided_slice/stack:output:09sequential_9/simple_rnn_18/strided_slice/stack_1:output:09sequential_9/simple_rnn_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_9/simple_rnn_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
'sequential_9/simple_rnn_18/zeros/packedPack1sequential_9/simple_rnn_18/strided_slice:output:02sequential_9/simple_rnn_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&sequential_9/simple_rnn_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 sequential_9/simple_rnn_18/zerosFill0sequential_9/simple_rnn_18/zeros/packed:output:0/sequential_9/simple_rnn_18/zeros/Const:output:0*
T0*'
_output_shapes
:?????????P~
)sequential_9/simple_rnn_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
$sequential_9/simple_rnn_18/transpose	Transposesimple_rnn_18_input2sequential_9/simple_rnn_18/transpose/perm:output:0*
T0*+
_output_shapes
:?????????z
"sequential_9/simple_rnn_18/Shape_1Shape(sequential_9/simple_rnn_18/transpose:y:0*
T0*
_output_shapes
:z
0sequential_9/simple_rnn_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_9/simple_rnn_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_9/simple_rnn_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_9/simple_rnn_18/strided_slice_1StridedSlice+sequential_9/simple_rnn_18/Shape_1:output:09sequential_9/simple_rnn_18/strided_slice_1/stack:output:0;sequential_9/simple_rnn_18/strided_slice_1/stack_1:output:0;sequential_9/simple_rnn_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
6sequential_9/simple_rnn_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
(sequential_9/simple_rnn_18/TensorArrayV2TensorListReserve?sequential_9/simple_rnn_18/TensorArrayV2/element_shape:output:03sequential_9/simple_rnn_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Psequential_9/simple_rnn_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Bsequential_9/simple_rnn_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_9/simple_rnn_18/transpose:y:0Ysequential_9/simple_rnn_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???z
0sequential_9/simple_rnn_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_9/simple_rnn_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_9/simple_rnn_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_9/simple_rnn_18/strided_slice_2StridedSlice(sequential_9/simple_rnn_18/transpose:y:09sequential_9/simple_rnn_18/strided_slice_2/stack:output:0;sequential_9/simple_rnn_18/strided_slice_2/stack_1:output:0;sequential_9/simple_rnn_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
Dsequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOpMsequential_9_simple_rnn_18_simple_rnn_cell_178_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
5sequential_9/simple_rnn_18/simple_rnn_cell_178/MatMulMatMul3sequential_9/simple_rnn_18/strided_slice_2:output:0Lsequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Esequential_9/simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOpNsequential_9_simple_rnn_18_simple_rnn_cell_178_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
6sequential_9/simple_rnn_18/simple_rnn_cell_178/BiasAddBiasAdd?sequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul:product:0Msequential_9/simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Fsequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOpOsequential_9_simple_rnn_18_simple_rnn_cell_178_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
7sequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul_1MatMul)sequential_9/simple_rnn_18/zeros:output:0Nsequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
2sequential_9/simple_rnn_18/simple_rnn_cell_178/addAddV2?sequential_9/simple_rnn_18/simple_rnn_cell_178/BiasAdd:output:0Asequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
3sequential_9/simple_rnn_18/simple_rnn_cell_178/TanhTanh6sequential_9/simple_rnn_18/simple_rnn_cell_178/add:z:0*
T0*'
_output_shapes
:?????????P?
8sequential_9/simple_rnn_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
*sequential_9/simple_rnn_18/TensorArrayV2_1TensorListReserveAsequential_9/simple_rnn_18/TensorArrayV2_1/element_shape:output:03sequential_9/simple_rnn_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???a
sequential_9/simple_rnn_18/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_9/simple_rnn_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
-sequential_9/simple_rnn_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
 sequential_9/simple_rnn_18/whileWhile6sequential_9/simple_rnn_18/while/loop_counter:output:0<sequential_9/simple_rnn_18/while/maximum_iterations:output:0(sequential_9/simple_rnn_18/time:output:03sequential_9/simple_rnn_18/TensorArrayV2_1:handle:0)sequential_9/simple_rnn_18/zeros:output:03sequential_9/simple_rnn_18/strided_slice_1:output:0Rsequential_9/simple_rnn_18/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_9_simple_rnn_18_simple_rnn_cell_178_matmul_readvariableop_resourceNsequential_9_simple_rnn_18_simple_rnn_cell_178_biasadd_readvariableop_resourceOsequential_9_simple_rnn_18_simple_rnn_cell_178_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *:
body2R0
.sequential_9_simple_rnn_18_while_body_11516228*:
cond2R0
.sequential_9_simple_rnn_18_while_cond_11516227*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
Ksequential_9/simple_rnn_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
=sequential_9/simple_rnn_18/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_9/simple_rnn_18/while:output:3Tsequential_9/simple_rnn_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0?
0sequential_9/simple_rnn_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????|
2sequential_9/simple_rnn_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_9/simple_rnn_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_9/simple_rnn_18/strided_slice_3StridedSliceFsequential_9/simple_rnn_18/TensorArrayV2Stack/TensorListStack:tensor:09sequential_9/simple_rnn_18/strided_slice_3/stack:output:0;sequential_9/simple_rnn_18/strided_slice_3/stack_1:output:0;sequential_9/simple_rnn_18/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
+sequential_9/simple_rnn_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
&sequential_9/simple_rnn_18/transpose_1	TransposeFsequential_9/simple_rnn_18/TensorArrayV2Stack/TensorListStack:tensor:04sequential_9/simple_rnn_18/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P?
 sequential_9/dropout_18/IdentityIdentity*sequential_9/simple_rnn_18/transpose_1:y:0*
T0*+
_output_shapes
:?????????Py
 sequential_9/simple_rnn_19/ShapeShape)sequential_9/dropout_18/Identity:output:0*
T0*
_output_shapes
:x
.sequential_9/simple_rnn_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_9/simple_rnn_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_9/simple_rnn_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential_9/simple_rnn_19/strided_sliceStridedSlice)sequential_9/simple_rnn_19/Shape:output:07sequential_9/simple_rnn_19/strided_slice/stack:output:09sequential_9/simple_rnn_19/strided_slice/stack_1:output:09sequential_9/simple_rnn_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_9/simple_rnn_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
'sequential_9/simple_rnn_19/zeros/packedPack1sequential_9/simple_rnn_19/strided_slice:output:02sequential_9/simple_rnn_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&sequential_9/simple_rnn_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 sequential_9/simple_rnn_19/zerosFill0sequential_9/simple_rnn_19/zeros/packed:output:0/sequential_9/simple_rnn_19/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d~
)sequential_9/simple_rnn_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
$sequential_9/simple_rnn_19/transpose	Transpose)sequential_9/dropout_18/Identity:output:02sequential_9/simple_rnn_19/transpose/perm:output:0*
T0*+
_output_shapes
:?????????Pz
"sequential_9/simple_rnn_19/Shape_1Shape(sequential_9/simple_rnn_19/transpose:y:0*
T0*
_output_shapes
:z
0sequential_9/simple_rnn_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_9/simple_rnn_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_9/simple_rnn_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_9/simple_rnn_19/strided_slice_1StridedSlice+sequential_9/simple_rnn_19/Shape_1:output:09sequential_9/simple_rnn_19/strided_slice_1/stack:output:0;sequential_9/simple_rnn_19/strided_slice_1/stack_1:output:0;sequential_9/simple_rnn_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
6sequential_9/simple_rnn_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
(sequential_9/simple_rnn_19/TensorArrayV2TensorListReserve?sequential_9/simple_rnn_19/TensorArrayV2/element_shape:output:03sequential_9/simple_rnn_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Psequential_9/simple_rnn_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
Bsequential_9/simple_rnn_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_9/simple_rnn_19/transpose:y:0Ysequential_9/simple_rnn_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???z
0sequential_9/simple_rnn_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_9/simple_rnn_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_9/simple_rnn_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_9/simple_rnn_19/strided_slice_2StridedSlice(sequential_9/simple_rnn_19/transpose:y:09sequential_9/simple_rnn_19/strided_slice_2/stack:output:0;sequential_9/simple_rnn_19/strided_slice_2/stack_1:output:0;sequential_9/simple_rnn_19/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
Dsequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOpMsequential_9_simple_rnn_19_simple_rnn_cell_179_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
5sequential_9/simple_rnn_19/simple_rnn_cell_179/MatMulMatMul3sequential_9/simple_rnn_19/strided_slice_2:output:0Lsequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Esequential_9/simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOpNsequential_9_simple_rnn_19_simple_rnn_cell_179_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
6sequential_9/simple_rnn_19/simple_rnn_cell_179/BiasAddBiasAdd?sequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul:product:0Msequential_9/simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Fsequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOpOsequential_9_simple_rnn_19_simple_rnn_cell_179_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
7sequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul_1MatMul)sequential_9/simple_rnn_19/zeros:output:0Nsequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
2sequential_9/simple_rnn_19/simple_rnn_cell_179/addAddV2?sequential_9/simple_rnn_19/simple_rnn_cell_179/BiasAdd:output:0Asequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
3sequential_9/simple_rnn_19/simple_rnn_cell_179/TanhTanh6sequential_9/simple_rnn_19/simple_rnn_cell_179/add:z:0*
T0*'
_output_shapes
:?????????d?
8sequential_9/simple_rnn_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
*sequential_9/simple_rnn_19/TensorArrayV2_1TensorListReserveAsequential_9/simple_rnn_19/TensorArrayV2_1/element_shape:output:03sequential_9/simple_rnn_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???a
sequential_9/simple_rnn_19/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_9/simple_rnn_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
-sequential_9/simple_rnn_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
 sequential_9/simple_rnn_19/whileWhile6sequential_9/simple_rnn_19/while/loop_counter:output:0<sequential_9/simple_rnn_19/while/maximum_iterations:output:0(sequential_9/simple_rnn_19/time:output:03sequential_9/simple_rnn_19/TensorArrayV2_1:handle:0)sequential_9/simple_rnn_19/zeros:output:03sequential_9/simple_rnn_19/strided_slice_1:output:0Rsequential_9/simple_rnn_19/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_9_simple_rnn_19_simple_rnn_cell_179_matmul_readvariableop_resourceNsequential_9_simple_rnn_19_simple_rnn_cell_179_biasadd_readvariableop_resourceOsequential_9_simple_rnn_19_simple_rnn_cell_179_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *:
body2R0
.sequential_9_simple_rnn_19_while_body_11516333*:
cond2R0
.sequential_9_simple_rnn_19_while_cond_11516332*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
Ksequential_9/simple_rnn_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
=sequential_9/simple_rnn_19/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_9/simple_rnn_19/while:output:3Tsequential_9/simple_rnn_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0?
0sequential_9/simple_rnn_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????|
2sequential_9/simple_rnn_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_9/simple_rnn_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_9/simple_rnn_19/strided_slice_3StridedSliceFsequential_9/simple_rnn_19/TensorArrayV2Stack/TensorListStack:tensor:09sequential_9/simple_rnn_19/strided_slice_3/stack:output:0;sequential_9/simple_rnn_19/strided_slice_3/stack_1:output:0;sequential_9/simple_rnn_19/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask?
+sequential_9/simple_rnn_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
&sequential_9/simple_rnn_19/transpose_1	TransposeFsequential_9/simple_rnn_19/TensorArrayV2Stack/TensorListStack:tensor:04sequential_9/simple_rnn_19/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d?
 sequential_9/dropout_19/IdentityIdentity3sequential_9/simple_rnn_19/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
*sequential_9/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_9_dense_9_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
sequential_9/dense_9/MatMulMatMul)sequential_9/dropout_19/Identity:output:02sequential_9/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_9/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_9_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_9/dense_9/BiasAddBiasAdd%sequential_9/dense_9/MatMul:product:03sequential_9/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_9/dense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^sequential_9/dense_9/BiasAdd/ReadVariableOp+^sequential_9/dense_9/MatMul/ReadVariableOpF^sequential_9/simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOpE^sequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOpG^sequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp!^sequential_9/simple_rnn_18/whileF^sequential_9/simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOpE^sequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOpG^sequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp!^sequential_9/simple_rnn_19/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2Z
+sequential_9/dense_9/BiasAdd/ReadVariableOp+sequential_9/dense_9/BiasAdd/ReadVariableOp2X
*sequential_9/dense_9/MatMul/ReadVariableOp*sequential_9/dense_9/MatMul/ReadVariableOp2?
Esequential_9/simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOpEsequential_9/simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOp2?
Dsequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOpDsequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOp2?
Fsequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOpFsequential_9/simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp2D
 sequential_9/simple_rnn_18/while sequential_9/simple_rnn_18/while2?
Esequential_9/simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOpEsequential_9/simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOp2?
Dsequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOpDsequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOp2?
Fsequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOpFsequential_9/simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp2D
 sequential_9/simple_rnn_19/while sequential_9/simple_rnn_19/while:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_18_input
?
I
-__inference_dropout_19_layer_call_fn_11519240

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_11517239`
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

?
!simple_rnn_18_while_cond_115178208
4simple_rnn_18_while_simple_rnn_18_while_loop_counter>
:simple_rnn_18_while_simple_rnn_18_while_maximum_iterations#
simple_rnn_18_while_placeholder%
!simple_rnn_18_while_placeholder_1%
!simple_rnn_18_while_placeholder_2:
6simple_rnn_18_while_less_simple_rnn_18_strided_slice_1R
Nsimple_rnn_18_while_simple_rnn_18_while_cond_11517820___redundant_placeholder0R
Nsimple_rnn_18_while_simple_rnn_18_while_cond_11517820___redundant_placeholder1R
Nsimple_rnn_18_while_simple_rnn_18_while_cond_11517820___redundant_placeholder2R
Nsimple_rnn_18_while_simple_rnn_18_while_cond_11517820___redundant_placeholder3 
simple_rnn_18_while_identity
?
simple_rnn_18/while/LessLesssimple_rnn_18_while_placeholder6simple_rnn_18_while_less_simple_rnn_18_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_18/while/IdentityIdentitysimple_rnn_18/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_18_while_identity%simple_rnn_18/while/Identity:output:0*(
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
?
.sequential_9_simple_rnn_18_while_cond_11516227R
Nsequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_while_loop_counterX
Tsequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_while_maximum_iterations0
,sequential_9_simple_rnn_18_while_placeholder2
.sequential_9_simple_rnn_18_while_placeholder_12
.sequential_9_simple_rnn_18_while_placeholder_2T
Psequential_9_simple_rnn_18_while_less_sequential_9_simple_rnn_18_strided_slice_1l
hsequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_while_cond_11516227___redundant_placeholder0l
hsequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_while_cond_11516227___redundant_placeholder1l
hsequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_while_cond_11516227___redundant_placeholder2l
hsequential_9_simple_rnn_18_while_sequential_9_simple_rnn_18_while_cond_11516227___redundant_placeholder3-
)sequential_9_simple_rnn_18_while_identity
?
%sequential_9/simple_rnn_18/while/LessLess,sequential_9_simple_rnn_18_while_placeholderPsequential_9_simple_rnn_18_while_less_sequential_9_simple_rnn_18_strided_slice_1*
T0*
_output_shapes
: ?
)sequential_9/simple_rnn_18/while/IdentityIdentity)sequential_9/simple_rnn_18/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_9_simple_rnn_18_while_identity2sequential_9/simple_rnn_18/while/Identity:output:0*(
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
while_cond_11518557
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11518557___redundant_placeholder06
2while_while_cond_11518557___redundant_placeholder16
2while_while_cond_11518557___redundant_placeholder26
2while_while_cond_11518557___redundant_placeholder3
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
?
?
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517706
simple_rnn_18_input(
simple_rnn_18_11517684:P$
simple_rnn_18_11517686:P(
simple_rnn_18_11517688:PP(
simple_rnn_19_11517692:Pd$
simple_rnn_19_11517694:d(
simple_rnn_19_11517696:dd"
dense_9_11517700:d
dense_9_11517702:
identity??dense_9/StatefulPartitionedCall?%simple_rnn_18/StatefulPartitionedCall?%simple_rnn_19/StatefulPartitionedCall?
%simple_rnn_18/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_18_inputsimple_rnn_18_11517684simple_rnn_18_11517686simple_rnn_18_11517688*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11517104?
dropout_18/PartitionedCallPartitionedCall.simple_rnn_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_11517117?
%simple_rnn_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0simple_rnn_19_11517692simple_rnn_19_11517694simple_rnn_19_11517696*
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11517226?
dropout_19/PartitionedCallPartitionedCall.simple_rnn_19/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_11517239?
dense_9/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_9_11517700dense_9_11517702*
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
E__inference_dense_9_layer_call_and_return_conditional_losses_11517251w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_9/StatefulPartitionedCall&^simple_rnn_18/StatefulPartitionedCall&^simple_rnn_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2N
%simple_rnn_18/StatefulPartitionedCall%simple_rnn_18/StatefulPartitionedCall2N
%simple_rnn_19/StatefulPartitionedCall%simple_rnn_19/StatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_18_input
?=
?
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11517431

inputsD
2simple_rnn_cell_179_matmul_readvariableop_resource:PdA
3simple_rnn_cell_179_biasadd_readvariableop_resource:dF
4simple_rnn_cell_179_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_179/BiasAdd/ReadVariableOp?)simple_rnn_cell_179/MatMul/ReadVariableOp?+simple_rnn_cell_179/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_179_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_179/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_179_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_179/BiasAddBiasAdd$simple_rnn_cell_179/MatMul:product:02simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_179_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_179/MatMul_1MatMulzeros:output:03simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_179/addAddV2$simple_rnn_cell_179/BiasAdd:output:0&simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_179/TanhTanhsimple_rnn_cell_179/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_179_matmul_readvariableop_resource3simple_rnn_cell_179_biasadd_readvariableop_resource4simple_rnn_cell_179_matmul_1_readvariableop_resource*
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
while_body_11517365*
condR
while_cond_11517364*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_179/BiasAdd/ReadVariableOp*^simple_rnn_cell_179/MatMul/ReadVariableOp,^simple_rnn_cell_179/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_179/BiasAdd/ReadVariableOp*simple_rnn_cell_179/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_179/MatMul/ReadVariableOp)simple_rnn_cell_179/MatMul/ReadVariableOp2Z
+simple_rnn_cell_179/MatMul_1/ReadVariableOp+simple_rnn_cell_179/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517641

inputs(
simple_rnn_18_11517619:P$
simple_rnn_18_11517621:P(
simple_rnn_18_11517623:PP(
simple_rnn_19_11517627:Pd$
simple_rnn_19_11517629:d(
simple_rnn_19_11517631:dd"
dense_9_11517635:d
dense_9_11517637:
identity??dense_9/StatefulPartitionedCall?"dropout_18/StatefulPartitionedCall?"dropout_19/StatefulPartitionedCall?%simple_rnn_18/StatefulPartitionedCall?%simple_rnn_19/StatefulPartitionedCall?
%simple_rnn_18/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_18_11517619simple_rnn_18_11517621simple_rnn_18_11517623*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11517584?
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_11517460?
%simple_rnn_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0simple_rnn_19_11517627simple_rnn_19_11517629simple_rnn_19_11517631*
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11517431?
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_19/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_11517307?
dense_9/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_9_11517635dense_9_11517637*
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
E__inference_dense_9_layer_call_and_return_conditional_losses_11517251w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_9/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall&^simple_rnn_18/StatefulPartitionedCall&^simple_rnn_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2N
%simple_rnn_18/StatefulPartitionedCall%simple_rnn_18/StatefulPartitionedCall2N
%simple_rnn_19/StatefulPartitionedCall%simple_rnn_19/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_11518449
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11518449___redundant_placeholder06
2while_while_cond_11518449___redundant_placeholder16
2while_while_cond_11518449___redundant_placeholder26
2while_while_cond_11518449___redundant_placeholder3
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
?=
?
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11517104

inputsD
2simple_rnn_cell_178_matmul_readvariableop_resource:PA
3simple_rnn_cell_178_biasadd_readvariableop_resource:PF
4simple_rnn_cell_178_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_178/BiasAdd/ReadVariableOp?)simple_rnn_cell_178/MatMul/ReadVariableOp?+simple_rnn_cell_178/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_178_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_178/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_178_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_178/BiasAddBiasAdd$simple_rnn_cell_178/MatMul:product:02simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_178_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_178/MatMul_1MatMulzeros:output:03simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_178/addAddV2$simple_rnn_cell_178/BiasAdd:output:0&simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_178/TanhTanhsimple_rnn_cell_178/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_178_matmul_readvariableop_resource3simple_rnn_cell_178_biasadd_readvariableop_resource4simple_rnn_cell_178_matmul_1_readvariableop_resource*
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
while_body_11517038*
condR
while_cond_11517037*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_178/BiasAdd/ReadVariableOp*^simple_rnn_cell_178/MatMul/ReadVariableOp,^simple_rnn_cell_178/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_178/BiasAdd/ReadVariableOp*simple_rnn_cell_178/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_178/MatMul/ReadVariableOp)simple_rnn_cell_178/MatMul/ReadVariableOp2Z
+simple_rnn_cell_178/MatMul_1/ReadVariableOp+simple_rnn_cell_178/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?I
?
!__inference__traced_save_11519521
file_prefix-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopF
Bsavev2_simple_rnn_18_simple_rnn_cell_18_kernel_read_readvariableopP
Lsavev2_simple_rnn_18_simple_rnn_cell_18_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_18_simple_rnn_cell_18_bias_read_readvariableopF
Bsavev2_simple_rnn_19_simple_rnn_cell_19_kernel_read_readvariableopP
Lsavev2_simple_rnn_19_simple_rnn_cell_19_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_19_simple_rnn_cell_19_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_18_simple_rnn_cell_18_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_18_simple_rnn_cell_18_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_18_simple_rnn_cell_18_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_19_simple_rnn_cell_19_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_19_simple_rnn_cell_19_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_19_simple_rnn_cell_19_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_18_simple_rnn_cell_18_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_18_simple_rnn_cell_18_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_18_simple_rnn_cell_18_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_19_simple_rnn_cell_19_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_19_simple_rnn_cell_19_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_19_simple_rnn_cell_19_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopBsavev2_simple_rnn_18_simple_rnn_cell_18_kernel_read_readvariableopLsavev2_simple_rnn_18_simple_rnn_cell_18_recurrent_kernel_read_readvariableop@savev2_simple_rnn_18_simple_rnn_cell_18_bias_read_readvariableopBsavev2_simple_rnn_19_simple_rnn_cell_19_kernel_read_readvariableopLsavev2_simple_rnn_19_simple_rnn_cell_19_recurrent_kernel_read_readvariableop@savev2_simple_rnn_19_simple_rnn_cell_19_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableopIsavev2_adam_simple_rnn_18_simple_rnn_cell_18_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_18_simple_rnn_cell_18_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_18_simple_rnn_cell_18_bias_m_read_readvariableopIsavev2_adam_simple_rnn_19_simple_rnn_cell_19_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_19_simple_rnn_cell_19_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_19_simple_rnn_cell_19_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableopIsavev2_adam_simple_rnn_18_simple_rnn_cell_18_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_18_simple_rnn_cell_18_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_18_simple_rnn_cell_18_bias_v_read_readvariableopIsavev2_adam_simple_rnn_19_simple_rnn_cell_19_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_19_simple_rnn_cell_19_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_19_simple_rnn_cell_19_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?

g
H__inference_dropout_18_layer_call_and_return_conditional_losses_11518759

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
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
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
:?????????Ps
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Pm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517258

inputs(
simple_rnn_18_11517105:P$
simple_rnn_18_11517107:P(
simple_rnn_18_11517109:PP(
simple_rnn_19_11517227:Pd$
simple_rnn_19_11517229:d(
simple_rnn_19_11517231:dd"
dense_9_11517252:d
dense_9_11517254:
identity??dense_9/StatefulPartitionedCall?%simple_rnn_18/StatefulPartitionedCall?%simple_rnn_19/StatefulPartitionedCall?
%simple_rnn_18/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_18_11517105simple_rnn_18_11517107simple_rnn_18_11517109*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11517104?
dropout_18/PartitionedCallPartitionedCall.simple_rnn_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_11517117?
%simple_rnn_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0simple_rnn_19_11517227simple_rnn_19_11517229simple_rnn_19_11517231*
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11517226?
dropout_19/PartitionedCallPartitionedCall.simple_rnn_19/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_11517239?
dense_9/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_9_11517252dense_9_11517254*
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
E__inference_dense_9_layer_call_and_return_conditional_losses_11517251w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_9/StatefulPartitionedCall&^simple_rnn_18/StatefulPartitionedCall&^simple_rnn_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2N
%simple_rnn_18/StatefulPartitionedCall%simple_rnn_18/StatefulPartitionedCall2N
%simple_rnn_19/StatefulPartitionedCall%simple_rnn_19/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_9_layer_call_fn_11517681
simple_rnn_18_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517641o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_18_input
?
?
while_cond_11516625
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11516625___redundant_placeholder06
2while_while_cond_11516625___redundant_placeholder16
2while_while_cond_11516625___redundant_placeholder26
2while_while_cond_11516625___redundant_placeholder3
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
?
f
H__inference_dropout_18_layer_call_and_return_conditional_losses_11518747

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11519343

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
?!
?
while_body_11516759
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_179_11516781_0:Pd2
$while_simple_rnn_cell_179_11516783_0:d6
$while_simple_rnn_cell_179_11516785_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_179_11516781:Pd0
"while_simple_rnn_cell_179_11516783:d4
"while_simple_rnn_cell_179_11516785:dd??1while/simple_rnn_cell_179/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
1while/simple_rnn_cell_179/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_179_11516781_0$while_simple_rnn_cell_179_11516783_0$while_simple_rnn_cell_179_11516785_0*
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
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11516746?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_179/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_179/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp2^while/simple_rnn_cell_179/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_179_11516781$while_simple_rnn_cell_179_11516781_0"J
"while_simple_rnn_cell_179_11516783$while_simple_rnn_cell_179_11516783_0"J
"while_simple_rnn_cell_179_11516785$while_simple_rnn_cell_179_11516785_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2f
1while/simple_rnn_cell_179/StatefulPartitionedCall1while/simple_rnn_cell_179/StatefulPartitionedCall: 
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
?:
?
!simple_rnn_19_while_body_115179268
4simple_rnn_19_while_simple_rnn_19_while_loop_counter>
:simple_rnn_19_while_simple_rnn_19_while_maximum_iterations#
simple_rnn_19_while_placeholder%
!simple_rnn_19_while_placeholder_1%
!simple_rnn_19_while_placeholder_27
3simple_rnn_19_while_simple_rnn_19_strided_slice_1_0s
osimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0Z
Hsimple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resource_0:PdW
Isimple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resource_0:d\
Jsimple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0:dd 
simple_rnn_19_while_identity"
simple_rnn_19_while_identity_1"
simple_rnn_19_while_identity_2"
simple_rnn_19_while_identity_3"
simple_rnn_19_while_identity_45
1simple_rnn_19_while_simple_rnn_19_strided_slice_1q
msimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensorX
Fsimple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resource:PdU
Gsimple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resource:dZ
Hsimple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resource:dd??>simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOp?=simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOp??simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp?
Esimple_rnn_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
7simple_rnn_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_19_while_placeholderNsimple_rnn_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
=simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOpHsimple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
.simple_rnn_19/while/simple_rnn_cell_179/MatMulMatMul>simple_rnn_19/while/TensorArrayV2Read/TensorListGetItem:item:0Esimple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
>simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOpIsimple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
/simple_rnn_19/while/simple_rnn_cell_179/BiasAddBiasAdd8simple_rnn_19/while/simple_rnn_cell_179/MatMul:product:0Fsimple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
?simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOpJsimple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
0simple_rnn_19/while/simple_rnn_cell_179/MatMul_1MatMul!simple_rnn_19_while_placeholder_2Gsimple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_19/while/simple_rnn_cell_179/addAddV28simple_rnn_19/while/simple_rnn_cell_179/BiasAdd:output:0:simple_rnn_19/while/simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
,simple_rnn_19/while/simple_rnn_cell_179/TanhTanh/simple_rnn_19/while/simple_rnn_cell_179/add:z:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_19_while_placeholder_1simple_rnn_19_while_placeholder0simple_rnn_19/while/simple_rnn_cell_179/Tanh:y:0*
_output_shapes
: *
element_dtype0:???[
simple_rnn_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_19/while/addAddV2simple_rnn_19_while_placeholder"simple_rnn_19/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_19/while/add_1AddV24simple_rnn_19_while_simple_rnn_19_while_loop_counter$simple_rnn_19/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_19/while/IdentityIdentitysimple_rnn_19/while/add_1:z:0^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_19/while/Identity_1Identity:simple_rnn_19_while_simple_rnn_19_while_maximum_iterations^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_19/while/Identity_2Identitysimple_rnn_19/while/add:z:0^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_19/while/Identity_3IdentityHsimple_rnn_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_19/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_19/while/Identity_4Identity0simple_rnn_19/while/simple_rnn_cell_179/Tanh:y:0^simple_rnn_19/while/NoOp*
T0*'
_output_shapes
:?????????d?
simple_rnn_19/while/NoOpNoOp?^simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOp>^simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOp@^simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_19_while_identity%simple_rnn_19/while/Identity:output:0"I
simple_rnn_19_while_identity_1'simple_rnn_19/while/Identity_1:output:0"I
simple_rnn_19_while_identity_2'simple_rnn_19/while/Identity_2:output:0"I
simple_rnn_19_while_identity_3'simple_rnn_19/while/Identity_3:output:0"I
simple_rnn_19_while_identity_4'simple_rnn_19/while/Identity_4:output:0"h
1simple_rnn_19_while_simple_rnn_19_strided_slice_13simple_rnn_19_while_simple_rnn_19_strided_slice_1_0"?
Gsimple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resourceIsimple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resource_0"?
Hsimple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resourceJsimple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0"?
Fsimple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resourceHsimple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resource_0"?
msimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensorosimple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2?
>simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOp>simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOp2~
=simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOp=simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOp2?
?simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp?simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp: 
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
while_cond_11516758
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11516758___redundant_placeholder06
2while_while_cond_11516758___redundant_placeholder16
2while_while_cond_11516758___redundant_placeholder26
2while_while_cond_11516758___redundant_placeholder3
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

?
6__inference_simple_rnn_cell_179_layer_call_fn_11519371

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
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11516866o
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
?-
?
while_body_11518953
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_179_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_179_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_179_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_179_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_179/MatMul/ReadVariableOp?1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_179_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_179/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_179/BiasAddBiasAdd*while/simple_rnn_cell_179/MatMul:product:08while/simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_179/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_179/addAddV2*while/simple_rnn_cell_179/BiasAdd:output:0,while/simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_179/TanhTanh!while/simple_rnn_cell_179/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_179/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_179/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_179/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_179/MatMul/ReadVariableOp2^while/simple_rnn_cell_179/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_179_biasadd_readvariableop_resource;while_simple_rnn_cell_179_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_179_matmul_1_readvariableop_resource<while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_179_matmul_readvariableop_resource:while_simple_rnn_cell_179_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp0while/simple_rnn_cell_179/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_179/MatMul/ReadVariableOp/while/simple_rnn_cell_179/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp1while/simple_rnn_cell_179/MatMul_1/ReadVariableOp: 
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
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11519127

inputsD
2simple_rnn_cell_179_matmul_readvariableop_resource:PdA
3simple_rnn_cell_179_biasadd_readvariableop_resource:dF
4simple_rnn_cell_179_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_179/BiasAdd/ReadVariableOp?)simple_rnn_cell_179/MatMul/ReadVariableOp?+simple_rnn_cell_179/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_179_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_179/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_179_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_179/BiasAddBiasAdd$simple_rnn_cell_179/MatMul:product:02simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_179_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_179/MatMul_1MatMulzeros:output:03simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_179/addAddV2$simple_rnn_cell_179/BiasAdd:output:0&simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_179/TanhTanhsimple_rnn_cell_179/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_179_matmul_readvariableop_resource3simple_rnn_cell_179_biasadd_readvariableop_resource4simple_rnn_cell_179_matmul_1_readvariableop_resource*
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
while_body_11519061*
condR
while_cond_11519060*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_179/BiasAdd/ReadVariableOp*^simple_rnn_cell_179/MatMul/ReadVariableOp,^simple_rnn_cell_179/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_179/BiasAdd/ReadVariableOp*simple_rnn_cell_179/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_179/MatMul/ReadVariableOp)simple_rnn_cell_179/MatMul/ReadVariableOp2Z
+simple_rnn_cell_179/MatMul_1/ReadVariableOp+simple_rnn_cell_179/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11516574

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
̫
?	
J__inference_sequential_9_layer_call_and_return_conditional_losses_11518233

inputsR
@simple_rnn_18_simple_rnn_cell_178_matmul_readvariableop_resource:PO
Asimple_rnn_18_simple_rnn_cell_178_biasadd_readvariableop_resource:PT
Bsimple_rnn_18_simple_rnn_cell_178_matmul_1_readvariableop_resource:PPR
@simple_rnn_19_simple_rnn_cell_179_matmul_readvariableop_resource:PdO
Asimple_rnn_19_simple_rnn_cell_179_biasadd_readvariableop_resource:dT
Bsimple_rnn_19_simple_rnn_cell_179_matmul_1_readvariableop_resource:dd8
&dense_9_matmul_readvariableop_resource:d5
'dense_9_biasadd_readvariableop_resource:
identity??dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?8simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOp?7simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOp?9simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp?simple_rnn_18/while?8simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOp?7simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOp?9simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp?simple_rnn_19/whileI
simple_rnn_18/ShapeShapeinputs*
T0*
_output_shapes
:k
!simple_rnn_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_18/strided_sliceStridedSlicesimple_rnn_18/Shape:output:0*simple_rnn_18/strided_slice/stack:output:0,simple_rnn_18/strided_slice/stack_1:output:0,simple_rnn_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
simple_rnn_18/zeros/packedPack$simple_rnn_18/strided_slice:output:0%simple_rnn_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_18/zerosFill#simple_rnn_18/zeros/packed:output:0"simple_rnn_18/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pq
simple_rnn_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_18/transpose	Transposeinputs%simple_rnn_18/transpose/perm:output:0*
T0*+
_output_shapes
:?????????`
simple_rnn_18/Shape_1Shapesimple_rnn_18/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_18/strided_slice_1StridedSlicesimple_rnn_18/Shape_1:output:0,simple_rnn_18/strided_slice_1/stack:output:0.simple_rnn_18/strided_slice_1/stack_1:output:0.simple_rnn_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_18/TensorArrayV2TensorListReserve2simple_rnn_18/TensorArrayV2/element_shape:output:0&simple_rnn_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Csimple_rnn_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
5simple_rnn_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_18/transpose:y:0Lsimple_rnn_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???m
#simple_rnn_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_18/strided_slice_2StridedSlicesimple_rnn_18/transpose:y:0,simple_rnn_18/strided_slice_2/stack:output:0.simple_rnn_18/strided_slice_2/stack_1:output:0.simple_rnn_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
7simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp@simple_rnn_18_simple_rnn_cell_178_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
(simple_rnn_18/simple_rnn_cell_178/MatMulMatMul&simple_rnn_18/strided_slice_2:output:0?simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOpAsimple_rnn_18_simple_rnn_cell_178_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
)simple_rnn_18/simple_rnn_cell_178/BiasAddBiasAdd2simple_rnn_18/simple_rnn_cell_178/MatMul:product:0@simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
9simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOpBsimple_rnn_18_simple_rnn_cell_178_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
*simple_rnn_18/simple_rnn_cell_178/MatMul_1MatMulsimple_rnn_18/zeros:output:0Asimple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
%simple_rnn_18/simple_rnn_cell_178/addAddV22simple_rnn_18/simple_rnn_cell_178/BiasAdd:output:04simple_rnn_18/simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
&simple_rnn_18/simple_rnn_cell_178/TanhTanh)simple_rnn_18/simple_rnn_cell_178/add:z:0*
T0*'
_output_shapes
:?????????P|
+simple_rnn_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
simple_rnn_18/TensorArrayV2_1TensorListReserve4simple_rnn_18/TensorArrayV2_1/element_shape:output:0&simple_rnn_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???T
simple_rnn_18/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 simple_rnn_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_18/whileWhile)simple_rnn_18/while/loop_counter:output:0/simple_rnn_18/while/maximum_iterations:output:0simple_rnn_18/time:output:0&simple_rnn_18/TensorArrayV2_1:handle:0simple_rnn_18/zeros:output:0&simple_rnn_18/strided_slice_1:output:0Esimple_rnn_18/TensorArrayUnstack/TensorListFromTensor:output_handle:0@simple_rnn_18_simple_rnn_cell_178_matmul_readvariableop_resourceAsimple_rnn_18_simple_rnn_cell_178_biasadd_readvariableop_resourceBsimple_rnn_18_simple_rnn_cell_178_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *-
body%R#
!simple_rnn_18_while_body_11518041*-
cond%R#
!simple_rnn_18_while_cond_11518040*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
>simple_rnn_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
0simple_rnn_18/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_18/while:output:3Gsimple_rnn_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0v
#simple_rnn_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%simple_rnn_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_18/strided_slice_3StridedSlice9simple_rnn_18/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_18/strided_slice_3/stack:output:0.simple_rnn_18/strided_slice_3/stack_1:output:0.simple_rnn_18/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_masks
simple_rnn_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_18/transpose_1	Transpose9simple_rnn_18/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_18/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P]
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_18/dropout/MulMulsimple_rnn_18/transpose_1:y:0!dropout_18/dropout/Const:output:0*
T0*+
_output_shapes
:?????????Pe
dropout_18/dropout/ShapeShapesimple_rnn_18/transpose_1:y:0*
T0*
_output_shapes
:?
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
dtype0f
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????P?
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????P?
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P_
simple_rnn_19/ShapeShapedropout_18/dropout/Mul_1:z:0*
T0*
_output_shapes
:k
!simple_rnn_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_19/strided_sliceStridedSlicesimple_rnn_19/Shape:output:0*simple_rnn_19/strided_slice/stack:output:0,simple_rnn_19/strided_slice/stack_1:output:0,simple_rnn_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
simple_rnn_19/zeros/packedPack$simple_rnn_19/strided_slice:output:0%simple_rnn_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_19/zerosFill#simple_rnn_19/zeros/packed:output:0"simple_rnn_19/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dq
simple_rnn_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_19/transpose	Transposedropout_18/dropout/Mul_1:z:0%simple_rnn_19/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P`
simple_rnn_19/Shape_1Shapesimple_rnn_19/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_19/strided_slice_1StridedSlicesimple_rnn_19/Shape_1:output:0,simple_rnn_19/strided_slice_1/stack:output:0.simple_rnn_19/strided_slice_1/stack_1:output:0.simple_rnn_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_19/TensorArrayV2TensorListReserve2simple_rnn_19/TensorArrayV2/element_shape:output:0&simple_rnn_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Csimple_rnn_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
5simple_rnn_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_19/transpose:y:0Lsimple_rnn_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???m
#simple_rnn_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_19/strided_slice_2StridedSlicesimple_rnn_19/transpose:y:0,simple_rnn_19/strided_slice_2/stack:output:0.simple_rnn_19/strided_slice_2/stack_1:output:0.simple_rnn_19/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
7simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOp@simple_rnn_19_simple_rnn_cell_179_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
(simple_rnn_19/simple_rnn_cell_179/MatMulMatMul&simple_rnn_19/strided_slice_2:output:0?simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOpAsimple_rnn_19_simple_rnn_cell_179_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
)simple_rnn_19/simple_rnn_cell_179/BiasAddBiasAdd2simple_rnn_19/simple_rnn_cell_179/MatMul:product:0@simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
9simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOpBsimple_rnn_19_simple_rnn_cell_179_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
*simple_rnn_19/simple_rnn_cell_179/MatMul_1MatMulsimple_rnn_19/zeros:output:0Asimple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
%simple_rnn_19/simple_rnn_cell_179/addAddV22simple_rnn_19/simple_rnn_cell_179/BiasAdd:output:04simple_rnn_19/simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
&simple_rnn_19/simple_rnn_cell_179/TanhTanh)simple_rnn_19/simple_rnn_cell_179/add:z:0*
T0*'
_output_shapes
:?????????d|
+simple_rnn_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
simple_rnn_19/TensorArrayV2_1TensorListReserve4simple_rnn_19/TensorArrayV2_1/element_shape:output:0&simple_rnn_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???T
simple_rnn_19/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 simple_rnn_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_19/whileWhile)simple_rnn_19/while/loop_counter:output:0/simple_rnn_19/while/maximum_iterations:output:0simple_rnn_19/time:output:0&simple_rnn_19/TensorArrayV2_1:handle:0simple_rnn_19/zeros:output:0&simple_rnn_19/strided_slice_1:output:0Esimple_rnn_19/TensorArrayUnstack/TensorListFromTensor:output_handle:0@simple_rnn_19_simple_rnn_cell_179_matmul_readvariableop_resourceAsimple_rnn_19_simple_rnn_cell_179_biasadd_readvariableop_resourceBsimple_rnn_19_simple_rnn_cell_179_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *-
body%R#
!simple_rnn_19_while_body_11518153*-
cond%R#
!simple_rnn_19_while_cond_11518152*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
>simple_rnn_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
0simple_rnn_19/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_19/while:output:3Gsimple_rnn_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0v
#simple_rnn_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%simple_rnn_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_19/strided_slice_3StridedSlice9simple_rnn_19/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_19/strided_slice_3/stack:output:0.simple_rnn_19/strided_slice_3/stack_1:output:0.simple_rnn_19/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_masks
simple_rnn_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_19/transpose_1	Transpose9simple_rnn_19/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_19/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d]
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_19/dropout/MulMul&simple_rnn_19/strided_slice_3:output:0!dropout_19/dropout/Const:output:0*
T0*'
_output_shapes
:?????????dn
dropout_19/dropout/ShapeShape&simple_rnn_19/strided_slice_3:output:0*
T0*
_output_shapes
:?
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0f
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_9/MatMulMatMuldropout_19/dropout/Mul_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp9^simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOp8^simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOp:^simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp^simple_rnn_18/while9^simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOp8^simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOp:^simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp^simple_rnn_19/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2t
8simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOp8simple_rnn_18/simple_rnn_cell_178/BiasAdd/ReadVariableOp2r
7simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOp7simple_rnn_18/simple_rnn_cell_178/MatMul/ReadVariableOp2v
9simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp9simple_rnn_18/simple_rnn_cell_178/MatMul_1/ReadVariableOp2*
simple_rnn_18/whilesimple_rnn_18/while2t
8simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOp8simple_rnn_19/simple_rnn_cell_179/BiasAdd/ReadVariableOp2r
7simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOp7simple_rnn_19/simple_rnn_cell_179/MatMul/ReadVariableOp2v
9simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp9simple_rnn_19/simple_rnn_cell_179/MatMul_1/ReadVariableOp2*
simple_rnn_19/whilesimple_rnn_19/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11519405

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
?
?
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11516454

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
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11516530

inputs.
simple_rnn_cell_178_11516455:P*
simple_rnn_cell_178_11516457:P.
simple_rnn_cell_178_11516459:PP
identity??+simple_rnn_cell_178/StatefulPartitionedCall?while;
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
+simple_rnn_cell_178/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_178_11516455simple_rnn_cell_178_11516457simple_rnn_cell_178_11516459*
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
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11516454n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_178_11516455simple_rnn_cell_178_11516457simple_rnn_cell_178_11516459*
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
while_body_11516467*
condR
while_cond_11516466*8
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
NoOpNoOp,^simple_rnn_cell_178/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+simple_rnn_cell_178/StatefulPartitionedCall+simple_rnn_cell_178/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_11518665
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11518665___redundant_placeholder06
2while_while_cond_11518665___redundant_placeholder16
2while_while_cond_11518665___redundant_placeholder26
2while_while_cond_11518665___redundant_placeholder3
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
?
*__inference_dense_9_layer_call_fn_11519271

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
E__inference_dense_9_layer_call_and_return_conditional_losses_11517251o
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
?=
?
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11518624

inputsD
2simple_rnn_cell_178_matmul_readvariableop_resource:PA
3simple_rnn_cell_178_biasadd_readvariableop_resource:PF
4simple_rnn_cell_178_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_178/BiasAdd/ReadVariableOp?)simple_rnn_cell_178/MatMul/ReadVariableOp?+simple_rnn_cell_178/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_178_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_178/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_178_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_178/BiasAddBiasAdd$simple_rnn_cell_178/MatMul:product:02simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_178_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_178/MatMul_1MatMulzeros:output:03simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_178/addAddV2$simple_rnn_cell_178/BiasAdd:output:0&simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_178/TanhTanhsimple_rnn_cell_178/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_178_matmul_readvariableop_resource3simple_rnn_cell_178_biasadd_readvariableop_resource4simple_rnn_cell_178_matmul_1_readvariableop_resource*
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
while_body_11518558*
condR
while_cond_11518557*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_178/BiasAdd/ReadVariableOp*^simple_rnn_cell_178/MatMul/ReadVariableOp,^simple_rnn_cell_178/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_178/BiasAdd/ReadVariableOp*simple_rnn_cell_178/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_178/MatMul/ReadVariableOp)simple_rnn_cell_178/MatMul/ReadVariableOp2Z
+simple_rnn_cell_178/MatMul_1/ReadVariableOp+simple_rnn_cell_178/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?F
?
.sequential_9_simple_rnn_19_while_body_11516333R
Nsequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_while_loop_counterX
Tsequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_while_maximum_iterations0
,sequential_9_simple_rnn_19_while_placeholder2
.sequential_9_simple_rnn_19_while_placeholder_12
.sequential_9_simple_rnn_19_while_placeholder_2Q
Msequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_strided_slice_1_0?
?sequential_9_simple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0g
Usequential_9_simple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resource_0:Pdd
Vsequential_9_simple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resource_0:di
Wsequential_9_simple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0:dd-
)sequential_9_simple_rnn_19_while_identity/
+sequential_9_simple_rnn_19_while_identity_1/
+sequential_9_simple_rnn_19_while_identity_2/
+sequential_9_simple_rnn_19_while_identity_3/
+sequential_9_simple_rnn_19_while_identity_4O
Ksequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_strided_slice_1?
?sequential_9_simple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_simple_rnn_19_tensorarrayunstack_tensorlistfromtensore
Ssequential_9_simple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resource:Pdb
Tsequential_9_simple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resource:dg
Usequential_9_simple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resource:dd??Ksequential_9/simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOp?Jsequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOp?Lsequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp?
Rsequential_9/simple_rnn_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
Dsequential_9/simple_rnn_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_9_simple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0,sequential_9_simple_rnn_19_while_placeholder[sequential_9/simple_rnn_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
Jsequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOpReadVariableOpUsequential_9_simple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
;sequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMulMatMulKsequential_9/simple_rnn_19/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Ksequential_9/simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOpReadVariableOpVsequential_9_simple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
<sequential_9/simple_rnn_19/while/simple_rnn_cell_179/BiasAddBiasAddEsequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul:product:0Ssequential_9/simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Lsequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOpReadVariableOpWsequential_9_simple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
=sequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul_1MatMul.sequential_9_simple_rnn_19_while_placeholder_2Tsequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8sequential_9/simple_rnn_19/while/simple_rnn_cell_179/addAddV2Esequential_9/simple_rnn_19/while/simple_rnn_cell_179/BiasAdd:output:0Gsequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
9sequential_9/simple_rnn_19/while/simple_rnn_cell_179/TanhTanh<sequential_9/simple_rnn_19/while/simple_rnn_cell_179/add:z:0*
T0*'
_output_shapes
:?????????d?
Esequential_9/simple_rnn_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_9_simple_rnn_19_while_placeholder_1,sequential_9_simple_rnn_19_while_placeholder=sequential_9/simple_rnn_19/while/simple_rnn_cell_179/Tanh:y:0*
_output_shapes
: *
element_dtype0:???h
&sequential_9/simple_rnn_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
$sequential_9/simple_rnn_19/while/addAddV2,sequential_9_simple_rnn_19_while_placeholder/sequential_9/simple_rnn_19/while/add/y:output:0*
T0*
_output_shapes
: j
(sequential_9/simple_rnn_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
&sequential_9/simple_rnn_19/while/add_1AddV2Nsequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_while_loop_counter1sequential_9/simple_rnn_19/while/add_1/y:output:0*
T0*
_output_shapes
: ?
)sequential_9/simple_rnn_19/while/IdentityIdentity*sequential_9/simple_rnn_19/while/add_1:z:0&^sequential_9/simple_rnn_19/while/NoOp*
T0*
_output_shapes
: ?
+sequential_9/simple_rnn_19/while/Identity_1IdentityTsequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_while_maximum_iterations&^sequential_9/simple_rnn_19/while/NoOp*
T0*
_output_shapes
: ?
+sequential_9/simple_rnn_19/while/Identity_2Identity(sequential_9/simple_rnn_19/while/add:z:0&^sequential_9/simple_rnn_19/while/NoOp*
T0*
_output_shapes
: ?
+sequential_9/simple_rnn_19/while/Identity_3IdentityUsequential_9/simple_rnn_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_9/simple_rnn_19/while/NoOp*
T0*
_output_shapes
: :????
+sequential_9/simple_rnn_19/while/Identity_4Identity=sequential_9/simple_rnn_19/while/simple_rnn_cell_179/Tanh:y:0&^sequential_9/simple_rnn_19/while/NoOp*
T0*'
_output_shapes
:?????????d?
%sequential_9/simple_rnn_19/while/NoOpNoOpL^sequential_9/simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOpK^sequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOpM^sequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_9_simple_rnn_19_while_identity2sequential_9/simple_rnn_19/while/Identity:output:0"c
+sequential_9_simple_rnn_19_while_identity_14sequential_9/simple_rnn_19/while/Identity_1:output:0"c
+sequential_9_simple_rnn_19_while_identity_24sequential_9/simple_rnn_19/while/Identity_2:output:0"c
+sequential_9_simple_rnn_19_while_identity_34sequential_9/simple_rnn_19/while/Identity_3:output:0"c
+sequential_9_simple_rnn_19_while_identity_44sequential_9/simple_rnn_19/while/Identity_4:output:0"?
Ksequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_strided_slice_1Msequential_9_simple_rnn_19_while_sequential_9_simple_rnn_19_strided_slice_1_0"?
Tsequential_9_simple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resourceVsequential_9_simple_rnn_19_while_simple_rnn_cell_179_biasadd_readvariableop_resource_0"?
Usequential_9_simple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resourceWsequential_9_simple_rnn_19_while_simple_rnn_cell_179_matmul_1_readvariableop_resource_0"?
Ssequential_9_simple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resourceUsequential_9_simple_rnn_19_while_simple_rnn_cell_179_matmul_readvariableop_resource_0"?
?sequential_9_simple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor?sequential_9_simple_rnn_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_simple_rnn_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2?
Ksequential_9/simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOpKsequential_9/simple_rnn_19/while/simple_rnn_cell_179/BiasAdd/ReadVariableOp2?
Jsequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOpJsequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul/ReadVariableOp2?
Lsequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOpLsequential_9/simple_rnn_19/while/simple_rnn_cell_179/MatMul_1/ReadVariableOp: 
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
?>
?
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11518408
inputs_0D
2simple_rnn_cell_178_matmul_readvariableop_resource:PA
3simple_rnn_cell_178_biasadd_readvariableop_resource:PF
4simple_rnn_cell_178_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_178/BiasAdd/ReadVariableOp?)simple_rnn_cell_178/MatMul/ReadVariableOp?+simple_rnn_cell_178/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_178_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_178/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_178_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_178/BiasAddBiasAdd$simple_rnn_cell_178/MatMul:product:02simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_178_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_178/MatMul_1MatMulzeros:output:03simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_178/addAddV2$simple_rnn_cell_178/BiasAdd:output:0&simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_178/TanhTanhsimple_rnn_cell_178/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_178_matmul_readvariableop_resource3simple_rnn_cell_178_biasadd_readvariableop_resource4simple_rnn_cell_178_matmul_1_readvariableop_resource*
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
while_body_11518342*
condR
while_cond_11518341*8
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
NoOpNoOp+^simple_rnn_cell_178/BiasAdd/ReadVariableOp*^simple_rnn_cell_178/MatMul/ReadVariableOp,^simple_rnn_cell_178/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2X
*simple_rnn_cell_178/BiasAdd/ReadVariableOp*simple_rnn_cell_178/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_178/MatMul/ReadVariableOp)simple_rnn_cell_178/MatMul/ReadVariableOp2Z
+simple_rnn_cell_178/MatMul_1/ReadVariableOp+simple_rnn_cell_178/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
0__inference_simple_rnn_18_layer_call_fn_11518289

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
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11517104s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_11517037
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11517037___redundant_placeholder06
2while_while_cond_11517037___redundant_placeholder16
2while_while_cond_11517037___redundant_placeholder26
2while_while_cond_11517037___redundant_placeholder3
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
?:
?
!simple_rnn_18_while_body_115180418
4simple_rnn_18_while_simple_rnn_18_while_loop_counter>
:simple_rnn_18_while_simple_rnn_18_while_maximum_iterations#
simple_rnn_18_while_placeholder%
!simple_rnn_18_while_placeholder_1%
!simple_rnn_18_while_placeholder_27
3simple_rnn_18_while_simple_rnn_18_strided_slice_1_0s
osimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0Z
Hsimple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resource_0:PW
Isimple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resource_0:P\
Jsimple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0:PP 
simple_rnn_18_while_identity"
simple_rnn_18_while_identity_1"
simple_rnn_18_while_identity_2"
simple_rnn_18_while_identity_3"
simple_rnn_18_while_identity_45
1simple_rnn_18_while_simple_rnn_18_strided_slice_1q
msimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensorX
Fsimple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resource:PU
Gsimple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resource:PZ
Hsimple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resource:PP??>simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOp?=simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOp??simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp?
Esimple_rnn_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
7simple_rnn_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_18_while_placeholderNsimple_rnn_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
=simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOpReadVariableOpHsimple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
.simple_rnn_18/while/simple_rnn_cell_178/MatMulMatMul>simple_rnn_18/while/TensorArrayV2Read/TensorListGetItem:item:0Esimple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
>simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOpReadVariableOpIsimple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
/simple_rnn_18/while/simple_rnn_cell_178/BiasAddBiasAdd8simple_rnn_18/while/simple_rnn_cell_178/MatMul:product:0Fsimple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
?simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOpReadVariableOpJsimple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
0simple_rnn_18/while/simple_rnn_cell_178/MatMul_1MatMul!simple_rnn_18_while_placeholder_2Gsimple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_18/while/simple_rnn_cell_178/addAddV28simple_rnn_18/while/simple_rnn_cell_178/BiasAdd:output:0:simple_rnn_18/while/simple_rnn_cell_178/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
,simple_rnn_18/while/simple_rnn_cell_178/TanhTanh/simple_rnn_18/while/simple_rnn_cell_178/add:z:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_18_while_placeholder_1simple_rnn_18_while_placeholder0simple_rnn_18/while/simple_rnn_cell_178/Tanh:y:0*
_output_shapes
: *
element_dtype0:???[
simple_rnn_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_18/while/addAddV2simple_rnn_18_while_placeholder"simple_rnn_18/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_18/while/add_1AddV24simple_rnn_18_while_simple_rnn_18_while_loop_counter$simple_rnn_18/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_18/while/IdentityIdentitysimple_rnn_18/while/add_1:z:0^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_18/while/Identity_1Identity:simple_rnn_18_while_simple_rnn_18_while_maximum_iterations^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_18/while/Identity_2Identitysimple_rnn_18/while/add:z:0^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_18/while/Identity_3IdentityHsimple_rnn_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_18/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_18/while/Identity_4Identity0simple_rnn_18/while/simple_rnn_cell_178/Tanh:y:0^simple_rnn_18/while/NoOp*
T0*'
_output_shapes
:?????????P?
simple_rnn_18/while/NoOpNoOp?^simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOp>^simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOp@^simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_18_while_identity%simple_rnn_18/while/Identity:output:0"I
simple_rnn_18_while_identity_1'simple_rnn_18/while/Identity_1:output:0"I
simple_rnn_18_while_identity_2'simple_rnn_18/while/Identity_2:output:0"I
simple_rnn_18_while_identity_3'simple_rnn_18/while/Identity_3:output:0"I
simple_rnn_18_while_identity_4'simple_rnn_18/while/Identity_4:output:0"h
1simple_rnn_18_while_simple_rnn_18_strided_slice_13simple_rnn_18_while_simple_rnn_18_strided_slice_1_0"?
Gsimple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resourceIsimple_rnn_18_while_simple_rnn_cell_178_biasadd_readvariableop_resource_0"?
Hsimple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resourceJsimple_rnn_18_while_simple_rnn_cell_178_matmul_1_readvariableop_resource_0"?
Fsimple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resourceHsimple_rnn_18_while_simple_rnn_cell_178_matmul_readvariableop_resource_0"?
msimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensorosimple_rnn_18_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2?
>simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOp>simple_rnn_18/while/simple_rnn_cell_178/BiasAdd/ReadVariableOp2~
=simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOp=simple_rnn_18/while/simple_rnn_cell_178/MatMul/ReadVariableOp2?
?simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp?simple_rnn_18/while/simple_rnn_cell_178/MatMul_1/ReadVariableOp: 
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
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11516981

inputs.
simple_rnn_cell_179_11516906:Pd*
simple_rnn_cell_179_11516908:d.
simple_rnn_cell_179_11516910:dd
identity??+simple_rnn_cell_179/StatefulPartitionedCall?while;
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
+simple_rnn_cell_179/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_179_11516906simple_rnn_cell_179_11516908simple_rnn_cell_179_11516910*
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
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11516866n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_179_11516906simple_rnn_cell_179_11516908simple_rnn_cell_179_11516910*
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
while_body_11516918*
condR
while_cond_11516917*8
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
NoOpNoOp,^simple_rnn_cell_179/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2Z
+simple_rnn_cell_179/StatefulPartitionedCall+simple_rnn_cell_179/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?

g
H__inference_dropout_18_layer_call_and_return_conditional_losses_11517460

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
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
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
:?????????Ps
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Pm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?	
?
/__inference_sequential_9_layer_call_fn_11517758

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
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517258o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_simple_rnn_18_layer_call_fn_11518300

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
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11517584s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
W
simple_rnn_18_input@
%serving_default_simple_rnn_18_input:0?????????;
dense_90
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
/__inference_sequential_9_layer_call_fn_11517277
/__inference_sequential_9_layer_call_fn_11517758
/__inference_sequential_9_layer_call_fn_11517779
/__inference_sequential_9_layer_call_fn_11517681?
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
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517999
J__inference_sequential_9_layer_call_and_return_conditional_losses_11518233
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517706
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517731?
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
#__inference__wrapped_model_11516406simple_rnn_18_input"?
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
0__inference_simple_rnn_18_layer_call_fn_11518267
0__inference_simple_rnn_18_layer_call_fn_11518278
0__inference_simple_rnn_18_layer_call_fn_11518289
0__inference_simple_rnn_18_layer_call_fn_11518300?
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
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11518408
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11518516
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11518624
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11518732?
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
-__inference_dropout_18_layer_call_fn_11518737
-__inference_dropout_18_layer_call_fn_11518742?
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
H__inference_dropout_18_layer_call_and_return_conditional_losses_11518747
H__inference_dropout_18_layer_call_and_return_conditional_losses_11518759?
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
0__inference_simple_rnn_19_layer_call_fn_11518770
0__inference_simple_rnn_19_layer_call_fn_11518781
0__inference_simple_rnn_19_layer_call_fn_11518792
0__inference_simple_rnn_19_layer_call_fn_11518803?
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
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11518911
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11519019
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11519127
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11519235?
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
-__inference_dropout_19_layer_call_fn_11519240
-__inference_dropout_19_layer_call_fn_11519245?
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
H__inference_dropout_19_layer_call_and_return_conditional_losses_11519250
H__inference_dropout_19_layer_call_and_return_conditional_losses_11519262?
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
 :d2dense_9/kernel
:2dense_9/bias
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
*__inference_dense_9_layer_call_fn_11519271?
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
E__inference_dense_9_layer_call_and_return_conditional_losses_11519281?
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
&__inference_signature_wrapper_11518256simple_rnn_18_input"?
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
9:7P2'simple_rnn_18/simple_rnn_cell_18/kernel
C:APP21simple_rnn_18/simple_rnn_cell_18/recurrent_kernel
3:1P2%simple_rnn_18/simple_rnn_cell_18/bias
9:7Pd2'simple_rnn_19/simple_rnn_cell_19/kernel
C:Add21simple_rnn_19/simple_rnn_cell_19/recurrent_kernel
3:1d2%simple_rnn_19/simple_rnn_cell_19/bias
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
6__inference_simple_rnn_cell_178_layer_call_fn_11519295
6__inference_simple_rnn_cell_178_layer_call_fn_11519309?
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
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11519326
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11519343?
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
6__inference_simple_rnn_cell_179_layer_call_fn_11519357
6__inference_simple_rnn_cell_179_layer_call_fn_11519371?
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
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11519388
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11519405?
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
%:#d2Adam/dense_9/kernel/m
:2Adam/dense_9/bias/m
>:<P2.Adam/simple_rnn_18/simple_rnn_cell_18/kernel/m
H:FPP28Adam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/m
8:6P2,Adam/simple_rnn_18/simple_rnn_cell_18/bias/m
>:<Pd2.Adam/simple_rnn_19/simple_rnn_cell_19/kernel/m
H:Fdd28Adam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/m
8:6d2,Adam/simple_rnn_19/simple_rnn_cell_19/bias/m
%:#d2Adam/dense_9/kernel/v
:2Adam/dense_9/bias/v
>:<P2.Adam/simple_rnn_18/simple_rnn_cell_18/kernel/v
H:FPP28Adam/simple_rnn_18/simple_rnn_cell_18/recurrent_kernel/v
8:6P2,Adam/simple_rnn_18/simple_rnn_cell_18/bias/v
>:<Pd2.Adam/simple_rnn_19/simple_rnn_cell_19/kernel/v
H:Fdd28Adam/simple_rnn_19/simple_rnn_cell_19/recurrent_kernel/v
8:6d2,Adam/simple_rnn_19/simple_rnn_cell_19/bias/v?
#__inference__wrapped_model_11516406ACBDFE23@?=
6?3
1?.
simple_rnn_18_input?????????
? "1?.
,
dense_9!?
dense_9??????????
E__inference_dense_9_layer_call_and_return_conditional_losses_11519281\23/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? }
*__inference_dense_9_layer_call_fn_11519271O23/?,
%?"
 ?
inputs?????????d
? "???????????
H__inference_dropout_18_layer_call_and_return_conditional_losses_11518747d7?4
-?*
$?!
inputs?????????P
p 
? ")?&
?
0?????????P
? ?
H__inference_dropout_18_layer_call_and_return_conditional_losses_11518759d7?4
-?*
$?!
inputs?????????P
p
? ")?&
?
0?????????P
? ?
-__inference_dropout_18_layer_call_fn_11518737W7?4
-?*
$?!
inputs?????????P
p 
? "??????????P?
-__inference_dropout_18_layer_call_fn_11518742W7?4
-?*
$?!
inputs?????????P
p
? "??????????P?
H__inference_dropout_19_layer_call_and_return_conditional_losses_11519250\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? ?
H__inference_dropout_19_layer_call_and_return_conditional_losses_11519262\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
-__inference_dropout_19_layer_call_fn_11519240O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
-__inference_dropout_19_layer_call_fn_11519245O3?0
)?&
 ?
inputs?????????d
p
? "??????????d?
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517706{ACBDFE23H?E
>?;
1?.
simple_rnn_18_input?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517731{ACBDFE23H?E
>?;
1?.
simple_rnn_18_input?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_9_layer_call_and_return_conditional_losses_11517999nACBDFE23;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_9_layer_call_and_return_conditional_losses_11518233nACBDFE23;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_9_layer_call_fn_11517277nACBDFE23H?E
>?;
1?.
simple_rnn_18_input?????????
p 

 
? "???????????
/__inference_sequential_9_layer_call_fn_11517681nACBDFE23H?E
>?;
1?.
simple_rnn_18_input?????????
p

 
? "???????????
/__inference_sequential_9_layer_call_fn_11517758aACBDFE23;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
/__inference_sequential_9_layer_call_fn_11517779aACBDFE23;?8
1?.
$?!
inputs?????????
p

 
? "???????????
&__inference_signature_wrapper_11518256?ACBDFE23W?T
? 
M?J
H
simple_rnn_18_input1?.
simple_rnn_18_input?????????"1?.
,
dense_9!?
dense_9??????????
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11518408?ACBO?L
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
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11518516?ACBO?L
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
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11518624qACB??<
5?2
$?!
inputs?????????

 
p 

 
? ")?&
?
0?????????P
? ?
K__inference_simple_rnn_18_layer_call_and_return_conditional_losses_11518732qACB??<
5?2
$?!
inputs?????????

 
p

 
? ")?&
?
0?????????P
? ?
0__inference_simple_rnn_18_layer_call_fn_11518267}ACBO?L
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
0__inference_simple_rnn_18_layer_call_fn_11518278}ACBO?L
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
0__inference_simple_rnn_18_layer_call_fn_11518289dACB??<
5?2
$?!
inputs?????????

 
p 

 
? "??????????P?
0__inference_simple_rnn_18_layer_call_fn_11518300dACB??<
5?2
$?!
inputs?????????

 
p

 
? "??????????P?
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11518911}DFEO?L
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
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11519019}DFEO?L
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
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11519127mDFE??<
5?2
$?!
inputs?????????P

 
p 

 
? "%?"
?
0?????????d
? ?
K__inference_simple_rnn_19_layer_call_and_return_conditional_losses_11519235mDFE??<
5?2
$?!
inputs?????????P

 
p

 
? "%?"
?
0?????????d
? ?
0__inference_simple_rnn_19_layer_call_fn_11518770pDFEO?L
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
0__inference_simple_rnn_19_layer_call_fn_11518781pDFEO?L
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
0__inference_simple_rnn_19_layer_call_fn_11518792`DFE??<
5?2
$?!
inputs?????????P

 
p 

 
? "??????????d?
0__inference_simple_rnn_19_layer_call_fn_11518803`DFE??<
5?2
$?!
inputs?????????P

 
p

 
? "??????????d?
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11519326?ACB\?Y
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
Q__inference_simple_rnn_cell_178_layer_call_and_return_conditional_losses_11519343?ACB\?Y
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
6__inference_simple_rnn_cell_178_layer_call_fn_11519295?ACB\?Y
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
6__inference_simple_rnn_cell_178_layer_call_fn_11519309?ACB\?Y
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
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11519388?DFE\?Y
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
Q__inference_simple_rnn_cell_179_layer_call_and_return_conditional_losses_11519405?DFE\?Y
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
6__inference_simple_rnn_cell_179_layer_call_fn_11519357?DFE\?Y
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
6__inference_simple_rnn_cell_179_layer_call_fn_11519371?DFE\?Y
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