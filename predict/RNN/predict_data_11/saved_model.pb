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
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:d*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
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
'simple_rnn_12/simple_rnn_cell_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*8
shared_name)'simple_rnn_12/simple_rnn_cell_12/kernel
?
;simple_rnn_12/simple_rnn_cell_12/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_12/simple_rnn_cell_12/kernel*
_output_shapes

:P*
dtype0
?
1simple_rnn_12/simple_rnn_cell_12/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*B
shared_name31simple_rnn_12/simple_rnn_cell_12/recurrent_kernel
?
Esimple_rnn_12/simple_rnn_cell_12/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_12/simple_rnn_cell_12/recurrent_kernel*
_output_shapes

:PP*
dtype0
?
%simple_rnn_12/simple_rnn_cell_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*6
shared_name'%simple_rnn_12/simple_rnn_cell_12/bias
?
9simple_rnn_12/simple_rnn_cell_12/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_12/simple_rnn_cell_12/bias*
_output_shapes
:P*
dtype0
?
'simple_rnn_13/simple_rnn_cell_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*8
shared_name)'simple_rnn_13/simple_rnn_cell_13/kernel
?
;simple_rnn_13/simple_rnn_cell_13/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_13/simple_rnn_cell_13/kernel*
_output_shapes

:Pd*
dtype0
?
1simple_rnn_13/simple_rnn_cell_13/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*B
shared_name31simple_rnn_13/simple_rnn_cell_13/recurrent_kernel
?
Esimple_rnn_13/simple_rnn_cell_13/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_13/simple_rnn_cell_13/recurrent_kernel*
_output_shapes

:dd*
dtype0
?
%simple_rnn_13/simple_rnn_cell_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*6
shared_name'%simple_rnn_13/simple_rnn_cell_13/bias
?
9simple_rnn_13/simple_rnn_cell_13/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_13/simple_rnn_cell_13/bias*
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
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:*
dtype0
?
.Adam/simple_rnn_12/simple_rnn_cell_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*?
shared_name0.Adam/simple_rnn_12/simple_rnn_cell_12/kernel/m
?
BAdam/simple_rnn_12/simple_rnn_cell_12/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_12/simple_rnn_cell_12/kernel/m*
_output_shapes

:P*
dtype0
?
8Adam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*I
shared_name:8Adam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/m
?
LAdam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/m*
_output_shapes

:PP*
dtype0
?
,Adam/simple_rnn_12/simple_rnn_cell_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*=
shared_name.,Adam/simple_rnn_12/simple_rnn_cell_12/bias/m
?
@Adam/simple_rnn_12/simple_rnn_cell_12/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_12/simple_rnn_cell_12/bias/m*
_output_shapes
:P*
dtype0
?
.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*?
shared_name0.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/m
?
BAdam/simple_rnn_13/simple_rnn_cell_13/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/m*
_output_shapes

:Pd*
dtype0
?
8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*I
shared_name:8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/m
?
LAdam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/m*
_output_shapes

:dd*
dtype0
?
,Adam/simple_rnn_13/simple_rnn_cell_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*=
shared_name.,Adam/simple_rnn_13/simple_rnn_cell_13/bias/m
?
@Adam/simple_rnn_13/simple_rnn_cell_13/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_13/simple_rnn_cell_13/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:*
dtype0
?
.Adam/simple_rnn_12/simple_rnn_cell_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*?
shared_name0.Adam/simple_rnn_12/simple_rnn_cell_12/kernel/v
?
BAdam/simple_rnn_12/simple_rnn_cell_12/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_12/simple_rnn_cell_12/kernel/v*
_output_shapes

:P*
dtype0
?
8Adam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*I
shared_name:8Adam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/v
?
LAdam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/v*
_output_shapes

:PP*
dtype0
?
,Adam/simple_rnn_12/simple_rnn_cell_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*=
shared_name.,Adam/simple_rnn_12/simple_rnn_cell_12/bias/v
?
@Adam/simple_rnn_12/simple_rnn_cell_12/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_12/simple_rnn_cell_12/bias/v*
_output_shapes
:P*
dtype0
?
.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*?
shared_name0.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/v
?
BAdam/simple_rnn_13/simple_rnn_cell_13/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/v*
_output_shapes

:Pd*
dtype0
?
8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*I
shared_name:8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/v
?
LAdam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/v*
_output_shapes

:dd*
dtype0
?
,Adam/simple_rnn_13/simple_rnn_cell_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*=
shared_name.,Adam/simple_rnn_13/simple_rnn_cell_13/bias/v
?
@Adam/simple_rnn_13/simple_rnn_cell_13/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_13/simple_rnn_cell_13/bias/v*
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
VARIABLE_VALUEdense_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUE'simple_rnn_12/simple_rnn_cell_12/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1simple_rnn_12/simple_rnn_cell_12/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_12/simple_rnn_cell_12/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'simple_rnn_13/simple_rnn_cell_13/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1simple_rnn_13/simple_rnn_cell_13/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_13/simple_rnn_cell_13/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/simple_rnn_12/simple_rnn_cell_12/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_12/simple_rnn_cell_12/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_13/simple_rnn_cell_13/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/simple_rnn_12/simple_rnn_cell_12/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_12/simple_rnn_cell_12/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_13/simple_rnn_cell_13/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
#serving_default_simple_rnn_12_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall#serving_default_simple_rnn_12_input'simple_rnn_12/simple_rnn_cell_12/kernel%simple_rnn_12/simple_rnn_cell_12/bias1simple_rnn_12/simple_rnn_cell_12/recurrent_kernel'simple_rnn_13/simple_rnn_cell_13/kernel%simple_rnn_13/simple_rnn_cell_13/bias1simple_rnn_13/simple_rnn_cell_13/recurrent_kerneldense_6/kerneldense_6/bias*
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
&__inference_signature_wrapper_11269067
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp;simple_rnn_12/simple_rnn_cell_12/kernel/Read/ReadVariableOpEsimple_rnn_12/simple_rnn_cell_12/recurrent_kernel/Read/ReadVariableOp9simple_rnn_12/simple_rnn_cell_12/bias/Read/ReadVariableOp;simple_rnn_13/simple_rnn_cell_13/kernel/Read/ReadVariableOpEsimple_rnn_13/simple_rnn_cell_13/recurrent_kernel/Read/ReadVariableOp9simple_rnn_13/simple_rnn_cell_13/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOpBAdam/simple_rnn_12/simple_rnn_cell_12/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_12/simple_rnn_cell_12/bias/m/Read/ReadVariableOpBAdam/simple_rnn_13/simple_rnn_cell_13/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_13/simple_rnn_cell_13/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOpBAdam/simple_rnn_12/simple_rnn_cell_12/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_12/simple_rnn_cell_12/bias/v/Read/ReadVariableOpBAdam/simple_rnn_13/simple_rnn_cell_13/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_13/simple_rnn_cell_13/bias/v/Read/ReadVariableOpConst*,
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
!__inference__traced_save_11270332
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate'simple_rnn_12/simple_rnn_cell_12/kernel1simple_rnn_12/simple_rnn_cell_12/recurrent_kernel%simple_rnn_12/simple_rnn_cell_12/bias'simple_rnn_13/simple_rnn_cell_13/kernel1simple_rnn_13/simple_rnn_cell_13/recurrent_kernel%simple_rnn_13/simple_rnn_cell_13/biastotalcountAdam/dense_6/kernel/mAdam/dense_6/bias/m.Adam/simple_rnn_12/simple_rnn_cell_12/kernel/m8Adam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/m,Adam/simple_rnn_12/simple_rnn_cell_12/bias/m.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/m8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/m,Adam/simple_rnn_13/simple_rnn_cell_13/bias/mAdam/dense_6/kernel/vAdam/dense_6/bias/v.Adam/simple_rnn_12/simple_rnn_cell_12/kernel/v8Adam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/v,Adam/simple_rnn_12/simple_rnn_cell_12/bias/v.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/v8Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/v,Adam/simple_rnn_13/simple_rnn_cell_13/bias/v*+
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
$__inference__traced_restore_11270435??
?
f
H__inference_dropout_12_layer_call_and_return_conditional_losses_11267928

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11270137

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

g
H__inference_dropout_12_layer_call_and_return_conditional_losses_11268271

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
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
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
:?????????Ps
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Pm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?=
?
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11267915

inputsD
2simple_rnn_cell_172_matmul_readvariableop_resource:PA
3simple_rnn_cell_172_biasadd_readvariableop_resource:PF
4simple_rnn_cell_172_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_172/BiasAdd/ReadVariableOp?)simple_rnn_cell_172/MatMul/ReadVariableOp?+simple_rnn_cell_172/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_172_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_172/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_172_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_172/BiasAddBiasAdd$simple_rnn_cell_172/MatMul:product:02simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_172_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_172/MatMul_1MatMulzeros:output:03simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_172/addAddV2$simple_rnn_cell_172/BiasAdd:output:0&simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_172/TanhTanhsimple_rnn_cell_172/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_172_matmul_readvariableop_resource3simple_rnn_cell_172_biasadd_readvariableop_resource4simple_rnn_cell_172_matmul_1_readvariableop_resource*
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
while_body_11267849*
condR
while_cond_11267848*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_172/BiasAdd/ReadVariableOp*^simple_rnn_cell_172/MatMul/ReadVariableOp,^simple_rnn_cell_172/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_172/BiasAdd/ReadVariableOp*simple_rnn_cell_172/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_172/MatMul/ReadVariableOp)simple_rnn_cell_172/MatMul/ReadVariableOp2Z
+simple_rnn_cell_172/MatMul_1/ReadVariableOp+simple_rnn_cell_172/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
while_body_11268176
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_173_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_173_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_173_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_173_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_173/MatMul/ReadVariableOp?1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_173_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_173/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_173/BiasAddBiasAdd*while/simple_rnn_cell_173/MatMul:product:08while/simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_173/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_173/addAddV2*while/simple_rnn_cell_173/BiasAdd:output:0,while/simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_173/TanhTanh!while/simple_rnn_cell_173/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_173/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_173/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_173/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_173/MatMul/ReadVariableOp2^while/simple_rnn_cell_173/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_173_biasadd_readvariableop_resource;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_173_matmul_1_readvariableop_resource<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_173_matmul_readvariableop_resource:while_simple_rnn_cell_173_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_173/MatMul/ReadVariableOp/while/simple_rnn_cell_173/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp: 
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
while_cond_11267436
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11267436___redundant_placeholder06
2while_while_cond_11267436___redundant_placeholder16
2while_while_cond_11267436___redundant_placeholder26
2while_while_cond_11267436___redundant_placeholder3
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
while_body_11269261
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_172_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_172_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_172_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_172_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_172/MatMul/ReadVariableOp?1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_172_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_172/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_172/BiasAddBiasAdd*while/simple_rnn_cell_172/MatMul:product:08while/simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_172/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_172/addAddV2*while/simple_rnn_cell_172/BiasAdd:output:0,while/simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_172/TanhTanh!while/simple_rnn_cell_172/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_172/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_172/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_172/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_172/MatMul/ReadVariableOp2^while/simple_rnn_cell_172/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_172_biasadd_readvariableop_resource;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_172_matmul_1_readvariableop_resource<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_172_matmul_readvariableop_resource:while_simple_rnn_cell_172_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_172/MatMul/ReadVariableOp/while/simple_rnn_cell_172/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp: 
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
?
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268517
simple_rnn_12_input(
simple_rnn_12_11268495:P$
simple_rnn_12_11268497:P(
simple_rnn_12_11268499:PP(
simple_rnn_13_11268503:Pd$
simple_rnn_13_11268505:d(
simple_rnn_13_11268507:dd"
dense_6_11268511:d
dense_6_11268513:
identity??dense_6/StatefulPartitionedCall?%simple_rnn_12/StatefulPartitionedCall?%simple_rnn_13/StatefulPartitionedCall?
%simple_rnn_12/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_12_inputsimple_rnn_12_11268495simple_rnn_12_11268497simple_rnn_12_11268499*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11267915?
dropout_12/PartitionedCallPartitionedCall.simple_rnn_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_11267928?
%simple_rnn_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0simple_rnn_13_11268503simple_rnn_13_11268505simple_rnn_13_11268507*
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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11268037?
dropout_13/PartitionedCallPartitionedCall.simple_rnn_13/StatefulPartitionedCall:output:0*
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
H__inference_dropout_13_layer_call_and_return_conditional_losses_11268050?
dense_6/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_6_11268511dense_6_11268513*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_11268062w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_6/StatefulPartitionedCall&^simple_rnn_12/StatefulPartitionedCall&^simple_rnn_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2N
%simple_rnn_12/StatefulPartitionedCall%simple_rnn_12/StatefulPartitionedCall2N
%simple_rnn_13/StatefulPartitionedCall%simple_rnn_13/StatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_12_input
?
f
-__inference_dropout_12_layer_call_fn_11269553

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
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_11268271s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?-
?
while_body_11267971
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_173_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_173_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_173_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_173_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_173/MatMul/ReadVariableOp?1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_173_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_173/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_173/BiasAddBiasAdd*while/simple_rnn_cell_173/MatMul:product:08while/simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_173/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_173/addAddV2*while/simple_rnn_cell_173/BiasAdd:output:0,while/simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_173/TanhTanh!while/simple_rnn_cell_173/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_173/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_173/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_173/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_173/MatMul/ReadVariableOp2^while/simple_rnn_cell_173/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_173_biasadd_readvariableop_resource;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_173_matmul_1_readvariableop_resource<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_173_matmul_readvariableop_resource:while_simple_rnn_cell_173_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_173/MatMul/ReadVariableOp/while/simple_rnn_cell_173/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp: 
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
while_body_11268329
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_172_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_172_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_172_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_172_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_172/MatMul/ReadVariableOp?1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_172_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_172/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_172/BiasAddBiasAdd*while/simple_rnn_cell_172/MatMul:product:08while/simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_172/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_172/addAddV2*while/simple_rnn_cell_172/BiasAdd:output:0,while/simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_172/TanhTanh!while/simple_rnn_cell_172/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_172/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_172/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_172/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_172/MatMul/ReadVariableOp2^while/simple_rnn_cell_172/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_172_biasadd_readvariableop_resource;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_172_matmul_1_readvariableop_resource<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_172_matmul_readvariableop_resource:while_simple_rnn_cell_172_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_172/MatMul/ReadVariableOp/while/simple_rnn_cell_172/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp: 
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
0__inference_simple_rnn_12_layer_call_fn_11269100

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
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11267915s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
!simple_rnn_12_while_cond_112688518
4simple_rnn_12_while_simple_rnn_12_while_loop_counter>
:simple_rnn_12_while_simple_rnn_12_while_maximum_iterations#
simple_rnn_12_while_placeholder%
!simple_rnn_12_while_placeholder_1%
!simple_rnn_12_while_placeholder_2:
6simple_rnn_12_while_less_simple_rnn_12_strided_slice_1R
Nsimple_rnn_12_while_simple_rnn_12_while_cond_11268851___redundant_placeholder0R
Nsimple_rnn_12_while_simple_rnn_12_while_cond_11268851___redundant_placeholder1R
Nsimple_rnn_12_while_simple_rnn_12_while_cond_11268851___redundant_placeholder2R
Nsimple_rnn_12_while_simple_rnn_12_while_cond_11268851___redundant_placeholder3 
simple_rnn_12_while_identity
?
simple_rnn_12/while/LessLesssimple_rnn_12_while_placeholder6simple_rnn_12_while_less_simple_rnn_12_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_12/while/IdentityIdentitysimple_rnn_12/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_12_while_identity%simple_rnn_12/while/Identity:output:0*(
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
/__inference_sequential_6_layer_call_fn_11268492
simple_rnn_12_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268452o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_12_input
?4
?
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11267500

inputs.
simple_rnn_cell_172_11267425:P*
simple_rnn_cell_172_11267427:P.
simple_rnn_cell_172_11267429:PP
identity??+simple_rnn_cell_172/StatefulPartitionedCall?while;
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
+simple_rnn_cell_172/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_172_11267425simple_rnn_cell_172_11267427simple_rnn_cell_172_11267429*
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
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11267385n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_172_11267425simple_rnn_cell_172_11267427simple_rnn_cell_172_11267429*
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
while_body_11267437*
condR
while_cond_11267436*8
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
NoOpNoOp,^simple_rnn_cell_172/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+simple_rnn_cell_172/StatefulPartitionedCall+simple_rnn_cell_172/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?F
?
.sequential_6_simple_rnn_12_while_body_11267039R
Nsequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_while_loop_counterX
Tsequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_while_maximum_iterations0
,sequential_6_simple_rnn_12_while_placeholder2
.sequential_6_simple_rnn_12_while_placeholder_12
.sequential_6_simple_rnn_12_while_placeholder_2Q
Msequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_strided_slice_1_0?
?sequential_6_simple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_12_tensorarrayunstack_tensorlistfromtensor_0g
Usequential_6_simple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resource_0:Pd
Vsequential_6_simple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resource_0:Pi
Wsequential_6_simple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0:PP-
)sequential_6_simple_rnn_12_while_identity/
+sequential_6_simple_rnn_12_while_identity_1/
+sequential_6_simple_rnn_12_while_identity_2/
+sequential_6_simple_rnn_12_while_identity_3/
+sequential_6_simple_rnn_12_while_identity_4O
Ksequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_strided_slice_1?
?sequential_6_simple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_12_tensorarrayunstack_tensorlistfromtensore
Ssequential_6_simple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resource:Pb
Tsequential_6_simple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resource:Pg
Usequential_6_simple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resource:PP??Ksequential_6/simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOp?Jsequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOp?Lsequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp?
Rsequential_6/simple_rnn_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Dsequential_6/simple_rnn_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_6_simple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_12_tensorarrayunstack_tensorlistfromtensor_0,sequential_6_simple_rnn_12_while_placeholder[sequential_6/simple_rnn_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
Jsequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOpUsequential_6_simple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
;sequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMulMatMulKsequential_6/simple_rnn_12/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Ksequential_6/simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOpVsequential_6_simple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
<sequential_6/simple_rnn_12/while/simple_rnn_cell_172/BiasAddBiasAddEsequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul:product:0Ssequential_6/simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Lsequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOpWsequential_6_simple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
=sequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul_1MatMul.sequential_6_simple_rnn_12_while_placeholder_2Tsequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8sequential_6/simple_rnn_12/while/simple_rnn_cell_172/addAddV2Esequential_6/simple_rnn_12/while/simple_rnn_cell_172/BiasAdd:output:0Gsequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
9sequential_6/simple_rnn_12/while/simple_rnn_cell_172/TanhTanh<sequential_6/simple_rnn_12/while/simple_rnn_cell_172/add:z:0*
T0*'
_output_shapes
:?????????P?
Esequential_6/simple_rnn_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_6_simple_rnn_12_while_placeholder_1,sequential_6_simple_rnn_12_while_placeholder=sequential_6/simple_rnn_12/while/simple_rnn_cell_172/Tanh:y:0*
_output_shapes
: *
element_dtype0:???h
&sequential_6/simple_rnn_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
$sequential_6/simple_rnn_12/while/addAddV2,sequential_6_simple_rnn_12_while_placeholder/sequential_6/simple_rnn_12/while/add/y:output:0*
T0*
_output_shapes
: j
(sequential_6/simple_rnn_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
&sequential_6/simple_rnn_12/while/add_1AddV2Nsequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_while_loop_counter1sequential_6/simple_rnn_12/while/add_1/y:output:0*
T0*
_output_shapes
: ?
)sequential_6/simple_rnn_12/while/IdentityIdentity*sequential_6/simple_rnn_12/while/add_1:z:0&^sequential_6/simple_rnn_12/while/NoOp*
T0*
_output_shapes
: ?
+sequential_6/simple_rnn_12/while/Identity_1IdentityTsequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_while_maximum_iterations&^sequential_6/simple_rnn_12/while/NoOp*
T0*
_output_shapes
: ?
+sequential_6/simple_rnn_12/while/Identity_2Identity(sequential_6/simple_rnn_12/while/add:z:0&^sequential_6/simple_rnn_12/while/NoOp*
T0*
_output_shapes
: ?
+sequential_6/simple_rnn_12/while/Identity_3IdentityUsequential_6/simple_rnn_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_6/simple_rnn_12/while/NoOp*
T0*
_output_shapes
: :????
+sequential_6/simple_rnn_12/while/Identity_4Identity=sequential_6/simple_rnn_12/while/simple_rnn_cell_172/Tanh:y:0&^sequential_6/simple_rnn_12/while/NoOp*
T0*'
_output_shapes
:?????????P?
%sequential_6/simple_rnn_12/while/NoOpNoOpL^sequential_6/simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOpK^sequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOpM^sequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_6_simple_rnn_12_while_identity2sequential_6/simple_rnn_12/while/Identity:output:0"c
+sequential_6_simple_rnn_12_while_identity_14sequential_6/simple_rnn_12/while/Identity_1:output:0"c
+sequential_6_simple_rnn_12_while_identity_24sequential_6/simple_rnn_12/while/Identity_2:output:0"c
+sequential_6_simple_rnn_12_while_identity_34sequential_6/simple_rnn_12/while/Identity_3:output:0"c
+sequential_6_simple_rnn_12_while_identity_44sequential_6/simple_rnn_12/while/Identity_4:output:0"?
Ksequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_strided_slice_1Msequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_strided_slice_1_0"?
Tsequential_6_simple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resourceVsequential_6_simple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resource_0"?
Usequential_6_simple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resourceWsequential_6_simple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0"?
Ssequential_6_simple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resourceUsequential_6_simple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resource_0"?
?sequential_6_simple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_12_tensorarrayunstack_tensorlistfromtensor?sequential_6_simple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2?
Ksequential_6/simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOpKsequential_6/simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOp2?
Jsequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOpJsequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOp2?
Lsequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOpLsequential_6/simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp: 
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
0__inference_simple_rnn_13_layer_call_fn_11269581
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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11267633o
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
0__inference_simple_rnn_12_layer_call_fn_11269078
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
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11267341|
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
while_body_11269764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_173_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_173_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_173_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_173_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_173/MatMul/ReadVariableOp?1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_173_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_173/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_173/BiasAddBiasAdd*while/simple_rnn_cell_173/MatMul:product:08while/simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_173/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_173/addAddV2*while/simple_rnn_cell_173/BiasAdd:output:0,while/simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_173/TanhTanh!while/simple_rnn_cell_173/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_173/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_173/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_173/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_173/MatMul/ReadVariableOp2^while/simple_rnn_cell_173/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_173_biasadd_readvariableop_resource;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_173_matmul_1_readvariableop_resource<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_173_matmul_readvariableop_resource:while_simple_rnn_cell_173_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_173/MatMul/ReadVariableOp/while/simple_rnn_cell_173/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp: 
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
g
H__inference_dropout_13_layer_call_and_return_conditional_losses_11270073

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
E__inference_dense_6_layer_call_and_return_conditional_losses_11270092

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
0__inference_simple_rnn_13_layer_call_fn_11269592
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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11267792o
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
?-
?
while_body_11269980
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_173_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_173_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_173_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_173_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_173/MatMul/ReadVariableOp?1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_173_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_173/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_173/BiasAddBiasAdd*while/simple_rnn_cell_173/MatMul:product:08while/simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_173/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_173/addAddV2*while/simple_rnn_cell_173/BiasAdd:output:0,while/simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_173/TanhTanh!while/simple_rnn_cell_173/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_173/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_173/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_173/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_173/MatMul/ReadVariableOp2^while/simple_rnn_cell_173/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_173_biasadd_readvariableop_resource;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_173_matmul_1_readvariableop_resource<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_173_matmul_readvariableop_resource:while_simple_rnn_cell_173_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_173/MatMul/ReadVariableOp/while/simple_rnn_cell_173/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp: 
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
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11269435

inputsD
2simple_rnn_cell_172_matmul_readvariableop_resource:PA
3simple_rnn_cell_172_biasadd_readvariableop_resource:PF
4simple_rnn_cell_172_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_172/BiasAdd/ReadVariableOp?)simple_rnn_cell_172/MatMul/ReadVariableOp?+simple_rnn_cell_172/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_172_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_172/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_172_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_172/BiasAddBiasAdd$simple_rnn_cell_172/MatMul:product:02simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_172_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_172/MatMul_1MatMulzeros:output:03simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_172/addAddV2$simple_rnn_cell_172/BiasAdd:output:0&simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_172/TanhTanhsimple_rnn_cell_172/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_172_matmul_readvariableop_resource3simple_rnn_cell_172_biasadd_readvariableop_resource4simple_rnn_cell_172_matmul_1_readvariableop_resource*
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
while_body_11269369*
condR
while_cond_11269368*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_172/BiasAdd/ReadVariableOp*^simple_rnn_cell_172/MatMul/ReadVariableOp,^simple_rnn_cell_172/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_172/BiasAdd/ReadVariableOp*simple_rnn_cell_172/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_172/MatMul/ReadVariableOp)simple_rnn_cell_172/MatMul/ReadVariableOp2Z
+simple_rnn_cell_172/MatMul_1/ReadVariableOp+simple_rnn_cell_172/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11270154

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
while_cond_11269763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11269763___redundant_placeholder06
2while_while_cond_11269763___redundant_placeholder16
2while_while_cond_11269763___redundant_placeholder26
2while_while_cond_11269763___redundant_placeholder3
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
?
I
-__inference_dropout_12_layer_call_fn_11269548

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
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_11267928d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
while_cond_11267970
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11267970___redundant_placeholder06
2while_while_cond_11267970___redundant_placeholder16
2while_while_cond_11267970___redundant_placeholder26
2while_while_cond_11267970___redundant_placeholder3
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
while_body_11267849
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_172_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_172_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_172_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_172_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_172/MatMul/ReadVariableOp?1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_172_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_172/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_172/BiasAddBiasAdd*while/simple_rnn_cell_172/MatMul:product:08while/simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_172/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_172/addAddV2*while/simple_rnn_cell_172/BiasAdd:output:0,while/simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_172/TanhTanh!while/simple_rnn_cell_172/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_172/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_172/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_172/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_172/MatMul/ReadVariableOp2^while/simple_rnn_cell_172/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_172_biasadd_readvariableop_resource;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_172_matmul_1_readvariableop_resource<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_172_matmul_readvariableop_resource:while_simple_rnn_cell_172_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_172/MatMul/ReadVariableOp/while/simple_rnn_cell_172/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp: 
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
ق
?
$__inference__traced_restore_11270435
file_prefix1
assignvariableop_dense_6_kernel:d-
assignvariableop_1_dense_6_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: L
:assignvariableop_7_simple_rnn_12_simple_rnn_cell_12_kernel:PV
Dassignvariableop_8_simple_rnn_12_simple_rnn_cell_12_recurrent_kernel:PPF
8assignvariableop_9_simple_rnn_12_simple_rnn_cell_12_bias:PM
;assignvariableop_10_simple_rnn_13_simple_rnn_cell_13_kernel:PdW
Eassignvariableop_11_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel:ddG
9assignvariableop_12_simple_rnn_13_simple_rnn_cell_13_bias:d#
assignvariableop_13_total: #
assignvariableop_14_count: ;
)assignvariableop_15_adam_dense_6_kernel_m:d5
'assignvariableop_16_adam_dense_6_bias_m:T
Bassignvariableop_17_adam_simple_rnn_12_simple_rnn_cell_12_kernel_m:P^
Lassignvariableop_18_adam_simple_rnn_12_simple_rnn_cell_12_recurrent_kernel_m:PPN
@assignvariableop_19_adam_simple_rnn_12_simple_rnn_cell_12_bias_m:PT
Bassignvariableop_20_adam_simple_rnn_13_simple_rnn_cell_13_kernel_m:Pd^
Lassignvariableop_21_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_m:ddN
@assignvariableop_22_adam_simple_rnn_13_simple_rnn_cell_13_bias_m:d;
)assignvariableop_23_adam_dense_6_kernel_v:d5
'assignvariableop_24_adam_dense_6_bias_v:T
Bassignvariableop_25_adam_simple_rnn_12_simple_rnn_cell_12_kernel_v:P^
Lassignvariableop_26_adam_simple_rnn_12_simple_rnn_cell_12_recurrent_kernel_v:PPN
@assignvariableop_27_adam_simple_rnn_12_simple_rnn_cell_12_bias_v:PT
Bassignvariableop_28_adam_simple_rnn_13_simple_rnn_cell_13_kernel_v:Pd^
Lassignvariableop_29_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_v:ddN
@assignvariableop_30_adam_simple_rnn_13_simple_rnn_cell_13_bias_v:d
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
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp:assignvariableop_7_simple_rnn_12_simple_rnn_cell_12_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpDassignvariableop_8_simple_rnn_12_simple_rnn_cell_12_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp8assignvariableop_9_simple_rnn_12_simple_rnn_cell_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp;assignvariableop_10_simple_rnn_13_simple_rnn_cell_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpEassignvariableop_11_simple_rnn_13_simple_rnn_cell_13_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp9assignvariableop_12_simple_rnn_13_simple_rnn_cell_13_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_6_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_6_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpBassignvariableop_17_adam_simple_rnn_12_simple_rnn_cell_12_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpLassignvariableop_18_adam_simple_rnn_12_simple_rnn_cell_12_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_simple_rnn_12_simple_rnn_cell_12_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpBassignvariableop_20_adam_simple_rnn_13_simple_rnn_cell_13_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpLassignvariableop_21_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp@assignvariableop_22_adam_simple_rnn_13_simple_rnn_cell_13_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_6_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_6_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpBassignvariableop_25_adam_simple_rnn_12_simple_rnn_cell_12_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpLassignvariableop_26_adam_simple_rnn_12_simple_rnn_cell_12_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_simple_rnn_12_simple_rnn_cell_12_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpBassignvariableop_28_adam_simple_rnn_13_simple_rnn_cell_13_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpLassignvariableop_29_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_simple_rnn_13_simple_rnn_cell_13_bias_vIdentity_30:output:0"/device:CPU:0*
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
?:
?
!simple_rnn_12_while_body_112686328
4simple_rnn_12_while_simple_rnn_12_while_loop_counter>
:simple_rnn_12_while_simple_rnn_12_while_maximum_iterations#
simple_rnn_12_while_placeholder%
!simple_rnn_12_while_placeholder_1%
!simple_rnn_12_while_placeholder_27
3simple_rnn_12_while_simple_rnn_12_strided_slice_1_0s
osimple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_12_tensorarrayunstack_tensorlistfromtensor_0Z
Hsimple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resource_0:PW
Isimple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resource_0:P\
Jsimple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0:PP 
simple_rnn_12_while_identity"
simple_rnn_12_while_identity_1"
simple_rnn_12_while_identity_2"
simple_rnn_12_while_identity_3"
simple_rnn_12_while_identity_45
1simple_rnn_12_while_simple_rnn_12_strided_slice_1q
msimple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_12_tensorarrayunstack_tensorlistfromtensorX
Fsimple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resource:PU
Gsimple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resource:PZ
Hsimple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resource:PP??>simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOp?=simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOp??simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp?
Esimple_rnn_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
7simple_rnn_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_12_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_12_while_placeholderNsimple_rnn_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
=simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOpHsimple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
.simple_rnn_12/while/simple_rnn_cell_172/MatMulMatMul>simple_rnn_12/while/TensorArrayV2Read/TensorListGetItem:item:0Esimple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
>simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOpIsimple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
/simple_rnn_12/while/simple_rnn_cell_172/BiasAddBiasAdd8simple_rnn_12/while/simple_rnn_cell_172/MatMul:product:0Fsimple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
?simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOpJsimple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
0simple_rnn_12/while/simple_rnn_cell_172/MatMul_1MatMul!simple_rnn_12_while_placeholder_2Gsimple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_12/while/simple_rnn_cell_172/addAddV28simple_rnn_12/while/simple_rnn_cell_172/BiasAdd:output:0:simple_rnn_12/while/simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
,simple_rnn_12/while/simple_rnn_cell_172/TanhTanh/simple_rnn_12/while/simple_rnn_cell_172/add:z:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_12_while_placeholder_1simple_rnn_12_while_placeholder0simple_rnn_12/while/simple_rnn_cell_172/Tanh:y:0*
_output_shapes
: *
element_dtype0:???[
simple_rnn_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_12/while/addAddV2simple_rnn_12_while_placeholder"simple_rnn_12/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_12/while/add_1AddV24simple_rnn_12_while_simple_rnn_12_while_loop_counter$simple_rnn_12/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_12/while/IdentityIdentitysimple_rnn_12/while/add_1:z:0^simple_rnn_12/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_12/while/Identity_1Identity:simple_rnn_12_while_simple_rnn_12_while_maximum_iterations^simple_rnn_12/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_12/while/Identity_2Identitysimple_rnn_12/while/add:z:0^simple_rnn_12/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_12/while/Identity_3IdentityHsimple_rnn_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_12/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_12/while/Identity_4Identity0simple_rnn_12/while/simple_rnn_cell_172/Tanh:y:0^simple_rnn_12/while/NoOp*
T0*'
_output_shapes
:?????????P?
simple_rnn_12/while/NoOpNoOp?^simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOp>^simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOp@^simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_12_while_identity%simple_rnn_12/while/Identity:output:0"I
simple_rnn_12_while_identity_1'simple_rnn_12/while/Identity_1:output:0"I
simple_rnn_12_while_identity_2'simple_rnn_12/while/Identity_2:output:0"I
simple_rnn_12_while_identity_3'simple_rnn_12/while/Identity_3:output:0"I
simple_rnn_12_while_identity_4'simple_rnn_12/while/Identity_4:output:0"h
1simple_rnn_12_while_simple_rnn_12_strided_slice_13simple_rnn_12_while_simple_rnn_12_strided_slice_1_0"?
Gsimple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resourceIsimple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resource_0"?
Hsimple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resourceJsimple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0"?
Fsimple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resourceHsimple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resource_0"?
msimple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_12_tensorarrayunstack_tensorlistfromtensorosimple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2?
>simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOp>simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOp2~
=simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOp=simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOp2?
?simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp?simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp: 
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
?
.sequential_6_simple_rnn_13_while_cond_11267143R
Nsequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_while_loop_counterX
Tsequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_while_maximum_iterations0
,sequential_6_simple_rnn_13_while_placeholder2
.sequential_6_simple_rnn_13_while_placeholder_12
.sequential_6_simple_rnn_13_while_placeholder_2T
Psequential_6_simple_rnn_13_while_less_sequential_6_simple_rnn_13_strided_slice_1l
hsequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_while_cond_11267143___redundant_placeholder0l
hsequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_while_cond_11267143___redundant_placeholder1l
hsequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_while_cond_11267143___redundant_placeholder2l
hsequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_while_cond_11267143___redundant_placeholder3-
)sequential_6_simple_rnn_13_while_identity
?
%sequential_6/simple_rnn_13/while/LessLess,sequential_6_simple_rnn_13_while_placeholderPsequential_6_simple_rnn_13_while_less_sequential_6_simple_rnn_13_strided_slice_1*
T0*
_output_shapes
: ?
)sequential_6/simple_rnn_13/while/IdentityIdentity)sequential_6/simple_rnn_13/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_6_simple_rnn_13_while_identity2sequential_6/simple_rnn_13/while/Identity:output:0*(
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
0__inference_simple_rnn_13_layer_call_fn_11269603

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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11268037o
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
:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?-
?
while_body_11269477
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_172_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_172_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_172_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_172_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_172/MatMul/ReadVariableOp?1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_172_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_172/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_172/BiasAddBiasAdd*while/simple_rnn_cell_172/MatMul:product:08while/simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_172/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_172/addAddV2*while/simple_rnn_cell_172/BiasAdd:output:0,while/simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_172/TanhTanh!while/simple_rnn_cell_172/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_172/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_172/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_172/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_172/MatMul/ReadVariableOp2^while/simple_rnn_cell_172/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_172_biasadd_readvariableop_resource;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_172_matmul_1_readvariableop_resource<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_172_matmul_readvariableop_resource:while_simple_rnn_cell_172_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_172/MatMul/ReadVariableOp/while/simple_rnn_cell_172/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp: 
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
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11270216

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
?>
?
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11269327
inputs_0D
2simple_rnn_cell_172_matmul_readvariableop_resource:PA
3simple_rnn_cell_172_biasadd_readvariableop_resource:PF
4simple_rnn_cell_172_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_172/BiasAdd/ReadVariableOp?)simple_rnn_cell_172/MatMul/ReadVariableOp?+simple_rnn_cell_172/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_172_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_172/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_172_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_172/BiasAddBiasAdd$simple_rnn_cell_172/MatMul:product:02simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_172_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_172/MatMul_1MatMulzeros:output:03simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_172/addAddV2$simple_rnn_cell_172/BiasAdd:output:0&simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_172/TanhTanhsimple_rnn_cell_172/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_172_matmul_readvariableop_resource3simple_rnn_cell_172_biasadd_readvariableop_resource4simple_rnn_cell_172_matmul_1_readvariableop_resource*
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
while_body_11269261*
condR
while_cond_11269260*8
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
NoOpNoOp+^simple_rnn_cell_172/BiasAdd/ReadVariableOp*^simple_rnn_cell_172/MatMul/ReadVariableOp,^simple_rnn_cell_172/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2X
*simple_rnn_cell_172/BiasAdd/ReadVariableOp*simple_rnn_cell_172/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_172/MatMul/ReadVariableOp)simple_rnn_cell_172/MatMul/ReadVariableOp2Z
+simple_rnn_cell_172/MatMul_1/ReadVariableOp+simple_rnn_cell_172/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?4
?
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11267633

inputs.
simple_rnn_cell_173_11267558:Pd*
simple_rnn_cell_173_11267560:d.
simple_rnn_cell_173_11267562:dd
identity??+simple_rnn_cell_173/StatefulPartitionedCall?while;
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
+simple_rnn_cell_173/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_173_11267558simple_rnn_cell_173_11267560simple_rnn_cell_173_11267562*
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
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11267557n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_173_11267558simple_rnn_cell_173_11267560simple_rnn_cell_173_11267562*
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
while_body_11267570*
condR
while_cond_11267569*8
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
NoOpNoOp,^simple_rnn_cell_173/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2Z
+simple_rnn_cell_173/StatefulPartitionedCall+simple_rnn_cell_173/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?
?
0__inference_simple_rnn_13_layer_call_fn_11269614

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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11268242o
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
:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
0__inference_simple_rnn_12_layer_call_fn_11269111

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
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11268395s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?=
?
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11268395

inputsD
2simple_rnn_cell_172_matmul_readvariableop_resource:PA
3simple_rnn_cell_172_biasadd_readvariableop_resource:PF
4simple_rnn_cell_172_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_172/BiasAdd/ReadVariableOp?)simple_rnn_cell_172/MatMul/ReadVariableOp?+simple_rnn_cell_172/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_172_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_172/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_172_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_172/BiasAddBiasAdd$simple_rnn_cell_172/MatMul:product:02simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_172_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_172/MatMul_1MatMulzeros:output:03simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_172/addAddV2$simple_rnn_cell_172/BiasAdd:output:0&simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_172/TanhTanhsimple_rnn_cell_172/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_172_matmul_readvariableop_resource3simple_rnn_cell_172_biasadd_readvariableop_resource4simple_rnn_cell_172_matmul_1_readvariableop_resource*
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
while_body_11268329*
condR
while_cond_11268328*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_172/BiasAdd/ReadVariableOp*^simple_rnn_cell_172/MatMul/ReadVariableOp,^simple_rnn_cell_172/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_172/BiasAdd/ReadVariableOp*simple_rnn_cell_172/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_172/MatMul/ReadVariableOp)simple_rnn_cell_172/MatMul/ReadVariableOp2Z
+simple_rnn_cell_172/MatMul_1/ReadVariableOp+simple_rnn_cell_172/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
!simple_rnn_12_while_cond_112686318
4simple_rnn_12_while_simple_rnn_12_while_loop_counter>
:simple_rnn_12_while_simple_rnn_12_while_maximum_iterations#
simple_rnn_12_while_placeholder%
!simple_rnn_12_while_placeholder_1%
!simple_rnn_12_while_placeholder_2:
6simple_rnn_12_while_less_simple_rnn_12_strided_slice_1R
Nsimple_rnn_12_while_simple_rnn_12_while_cond_11268631___redundant_placeholder0R
Nsimple_rnn_12_while_simple_rnn_12_while_cond_11268631___redundant_placeholder1R
Nsimple_rnn_12_while_simple_rnn_12_while_cond_11268631___redundant_placeholder2R
Nsimple_rnn_12_while_simple_rnn_12_while_cond_11268631___redundant_placeholder3 
simple_rnn_12_while_identity
?
simple_rnn_12/while/LessLesssimple_rnn_12_while_placeholder6simple_rnn_12_while_less_simple_rnn_12_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_12/while/IdentityIdentitysimple_rnn_12/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_12_while_identity%simple_rnn_12/while/Identity:output:0*(
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

?
6__inference_simple_rnn_cell_172_layer_call_fn_11270106

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
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11267265o
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
?4
?
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11267341

inputs.
simple_rnn_cell_172_11267266:P*
simple_rnn_cell_172_11267268:P.
simple_rnn_cell_172_11267270:PP
identity??+simple_rnn_cell_172/StatefulPartitionedCall?while;
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
+simple_rnn_cell_172/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_172_11267266simple_rnn_cell_172_11267268simple_rnn_cell_172_11267270*
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
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11267265n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_172_11267266simple_rnn_cell_172_11267268simple_rnn_cell_172_11267270*
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
while_body_11267278*
condR
while_cond_11267277*8
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
NoOpNoOp,^simple_rnn_cell_172/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+simple_rnn_cell_172/StatefulPartitionedCall+simple_rnn_cell_172/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?!
?
while_body_11267729
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_173_11267751_0:Pd2
$while_simple_rnn_cell_173_11267753_0:d6
$while_simple_rnn_cell_173_11267755_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_173_11267751:Pd0
"while_simple_rnn_cell_173_11267753:d4
"while_simple_rnn_cell_173_11267755:dd??1while/simple_rnn_cell_173/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
1while/simple_rnn_cell_173/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_173_11267751_0$while_simple_rnn_cell_173_11267753_0$while_simple_rnn_cell_173_11267755_0*
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
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11267677?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_173/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_173/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp2^while/simple_rnn_cell_173/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_173_11267751$while_simple_rnn_cell_173_11267751_0"J
"while_simple_rnn_cell_173_11267753$while_simple_rnn_cell_173_11267753_0"J
"while_simple_rnn_cell_173_11267755$while_simple_rnn_cell_173_11267755_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2f
1while/simple_rnn_cell_173/StatefulPartitionedCall1while/simple_rnn_cell_173/StatefulPartitionedCall: 
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
E__inference_dense_6_layer_call_and_return_conditional_losses_11268062

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
?
&__inference_signature_wrapper_11269067
simple_rnn_12_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_11267217o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_12_input
?:
?
!simple_rnn_13_while_body_112689648
4simple_rnn_13_while_simple_rnn_13_while_loop_counter>
:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations#
simple_rnn_13_while_placeholder%
!simple_rnn_13_while_placeholder_1%
!simple_rnn_13_while_placeholder_27
3simple_rnn_13_while_simple_rnn_13_strided_slice_1_0s
osimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0Z
Hsimple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resource_0:PdW
Isimple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resource_0:d\
Jsimple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0:dd 
simple_rnn_13_while_identity"
simple_rnn_13_while_identity_1"
simple_rnn_13_while_identity_2"
simple_rnn_13_while_identity_3"
simple_rnn_13_while_identity_45
1simple_rnn_13_while_simple_rnn_13_strided_slice_1q
msimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorX
Fsimple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resource:PdU
Gsimple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resource:dZ
Hsimple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resource:dd??>simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOp?=simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOp??simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp?
Esimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
7simple_rnn_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_13_while_placeholderNsimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
=simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOpHsimple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
.simple_rnn_13/while/simple_rnn_cell_173/MatMulMatMul>simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem:item:0Esimple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
>simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOpIsimple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
/simple_rnn_13/while/simple_rnn_cell_173/BiasAddBiasAdd8simple_rnn_13/while/simple_rnn_cell_173/MatMul:product:0Fsimple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
?simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOpJsimple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
0simple_rnn_13/while/simple_rnn_cell_173/MatMul_1MatMul!simple_rnn_13_while_placeholder_2Gsimple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_13/while/simple_rnn_cell_173/addAddV28simple_rnn_13/while/simple_rnn_cell_173/BiasAdd:output:0:simple_rnn_13/while/simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
,simple_rnn_13/while/simple_rnn_cell_173/TanhTanh/simple_rnn_13/while/simple_rnn_cell_173/add:z:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_13_while_placeholder_1simple_rnn_13_while_placeholder0simple_rnn_13/while/simple_rnn_cell_173/Tanh:y:0*
_output_shapes
: *
element_dtype0:???[
simple_rnn_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_13/while/addAddV2simple_rnn_13_while_placeholder"simple_rnn_13/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_13/while/add_1AddV24simple_rnn_13_while_simple_rnn_13_while_loop_counter$simple_rnn_13/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_13/while/IdentityIdentitysimple_rnn_13/while/add_1:z:0^simple_rnn_13/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_13/while/Identity_1Identity:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations^simple_rnn_13/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_13/while/Identity_2Identitysimple_rnn_13/while/add:z:0^simple_rnn_13/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_13/while/Identity_3IdentityHsimple_rnn_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_13/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_13/while/Identity_4Identity0simple_rnn_13/while/simple_rnn_cell_173/Tanh:y:0^simple_rnn_13/while/NoOp*
T0*'
_output_shapes
:?????????d?
simple_rnn_13/while/NoOpNoOp?^simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOp>^simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOp@^simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_13_while_identity%simple_rnn_13/while/Identity:output:0"I
simple_rnn_13_while_identity_1'simple_rnn_13/while/Identity_1:output:0"I
simple_rnn_13_while_identity_2'simple_rnn_13/while/Identity_2:output:0"I
simple_rnn_13_while_identity_3'simple_rnn_13/while/Identity_3:output:0"I
simple_rnn_13_while_identity_4'simple_rnn_13/while/Identity_4:output:0"h
1simple_rnn_13_while_simple_rnn_13_strided_slice_13simple_rnn_13_while_simple_rnn_13_strided_slice_1_0"?
Gsimple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resourceIsimple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resource_0"?
Hsimple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resourceJsimple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0"?
Fsimple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resourceHsimple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resource_0"?
msimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorosimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2?
>simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOp>simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOp2~
=simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOp=simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOp2?
?simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp?simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp: 
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
while_cond_11269979
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11269979___redundant_placeholder06
2while_while_cond_11269979___redundant_placeholder16
2while_while_cond_11269979___redundant_placeholder26
2while_while_cond_11269979___redundant_placeholder3
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
6__inference_simple_rnn_cell_173_layer_call_fn_11270182

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
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11267677o
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
?
?
while_cond_11269655
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11269655___redundant_placeholder06
2while_while_cond_11269655___redundant_placeholder16
2while_while_cond_11269655___redundant_placeholder26
2while_while_cond_11269655___redundant_placeholder3
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

g
H__inference_dropout_12_layer_call_and_return_conditional_losses_11269570

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
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
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
:?????????Ps
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Pm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?F
?
.sequential_6_simple_rnn_13_while_body_11267144R
Nsequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_while_loop_counterX
Tsequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_while_maximum_iterations0
,sequential_6_simple_rnn_13_while_placeholder2
.sequential_6_simple_rnn_13_while_placeholder_12
.sequential_6_simple_rnn_13_while_placeholder_2Q
Msequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_strided_slice_1_0?
?sequential_6_simple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0g
Usequential_6_simple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resource_0:Pdd
Vsequential_6_simple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resource_0:di
Wsequential_6_simple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0:dd-
)sequential_6_simple_rnn_13_while_identity/
+sequential_6_simple_rnn_13_while_identity_1/
+sequential_6_simple_rnn_13_while_identity_2/
+sequential_6_simple_rnn_13_while_identity_3/
+sequential_6_simple_rnn_13_while_identity_4O
Ksequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_strided_slice_1?
?sequential_6_simple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_13_tensorarrayunstack_tensorlistfromtensore
Ssequential_6_simple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resource:Pdb
Tsequential_6_simple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resource:dg
Usequential_6_simple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resource:dd??Ksequential_6/simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOp?Jsequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOp?Lsequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp?
Rsequential_6/simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
Dsequential_6/simple_rnn_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_6_simple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0,sequential_6_simple_rnn_13_while_placeholder[sequential_6/simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
Jsequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOpUsequential_6_simple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
;sequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMulMatMulKsequential_6/simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Ksequential_6/simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOpVsequential_6_simple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
<sequential_6/simple_rnn_13/while/simple_rnn_cell_173/BiasAddBiasAddEsequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul:product:0Ssequential_6/simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Lsequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOpWsequential_6_simple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
=sequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul_1MatMul.sequential_6_simple_rnn_13_while_placeholder_2Tsequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8sequential_6/simple_rnn_13/while/simple_rnn_cell_173/addAddV2Esequential_6/simple_rnn_13/while/simple_rnn_cell_173/BiasAdd:output:0Gsequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
9sequential_6/simple_rnn_13/while/simple_rnn_cell_173/TanhTanh<sequential_6/simple_rnn_13/while/simple_rnn_cell_173/add:z:0*
T0*'
_output_shapes
:?????????d?
Esequential_6/simple_rnn_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_6_simple_rnn_13_while_placeholder_1,sequential_6_simple_rnn_13_while_placeholder=sequential_6/simple_rnn_13/while/simple_rnn_cell_173/Tanh:y:0*
_output_shapes
: *
element_dtype0:???h
&sequential_6/simple_rnn_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
$sequential_6/simple_rnn_13/while/addAddV2,sequential_6_simple_rnn_13_while_placeholder/sequential_6/simple_rnn_13/while/add/y:output:0*
T0*
_output_shapes
: j
(sequential_6/simple_rnn_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
&sequential_6/simple_rnn_13/while/add_1AddV2Nsequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_while_loop_counter1sequential_6/simple_rnn_13/while/add_1/y:output:0*
T0*
_output_shapes
: ?
)sequential_6/simple_rnn_13/while/IdentityIdentity*sequential_6/simple_rnn_13/while/add_1:z:0&^sequential_6/simple_rnn_13/while/NoOp*
T0*
_output_shapes
: ?
+sequential_6/simple_rnn_13/while/Identity_1IdentityTsequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_while_maximum_iterations&^sequential_6/simple_rnn_13/while/NoOp*
T0*
_output_shapes
: ?
+sequential_6/simple_rnn_13/while/Identity_2Identity(sequential_6/simple_rnn_13/while/add:z:0&^sequential_6/simple_rnn_13/while/NoOp*
T0*
_output_shapes
: ?
+sequential_6/simple_rnn_13/while/Identity_3IdentityUsequential_6/simple_rnn_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_6/simple_rnn_13/while/NoOp*
T0*
_output_shapes
: :????
+sequential_6/simple_rnn_13/while/Identity_4Identity=sequential_6/simple_rnn_13/while/simple_rnn_cell_173/Tanh:y:0&^sequential_6/simple_rnn_13/while/NoOp*
T0*'
_output_shapes
:?????????d?
%sequential_6/simple_rnn_13/while/NoOpNoOpL^sequential_6/simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOpK^sequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOpM^sequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_6_simple_rnn_13_while_identity2sequential_6/simple_rnn_13/while/Identity:output:0"c
+sequential_6_simple_rnn_13_while_identity_14sequential_6/simple_rnn_13/while/Identity_1:output:0"c
+sequential_6_simple_rnn_13_while_identity_24sequential_6/simple_rnn_13/while/Identity_2:output:0"c
+sequential_6_simple_rnn_13_while_identity_34sequential_6/simple_rnn_13/while/Identity_3:output:0"c
+sequential_6_simple_rnn_13_while_identity_44sequential_6/simple_rnn_13/while/Identity_4:output:0"?
Ksequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_strided_slice_1Msequential_6_simple_rnn_13_while_sequential_6_simple_rnn_13_strided_slice_1_0"?
Tsequential_6_simple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resourceVsequential_6_simple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resource_0"?
Usequential_6_simple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resourceWsequential_6_simple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0"?
Ssequential_6_simple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resourceUsequential_6_simple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resource_0"?
?sequential_6_simple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor?sequential_6_simple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2?
Ksequential_6/simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOpKsequential_6/simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOp2?
Jsequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOpJsequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOp2?
Lsequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOpLsequential_6/simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp: 
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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11270046

inputsD
2simple_rnn_cell_173_matmul_readvariableop_resource:PdA
3simple_rnn_cell_173_biasadd_readvariableop_resource:dF
4simple_rnn_cell_173_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_173/BiasAdd/ReadVariableOp?)simple_rnn_cell_173/MatMul/ReadVariableOp?+simple_rnn_cell_173/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_173_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_173/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_173_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_173/BiasAddBiasAdd$simple_rnn_cell_173/MatMul:product:02simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_173_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_173/MatMul_1MatMulzeros:output:03simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_173/addAddV2$simple_rnn_cell_173/BiasAdd:output:0&simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_173/TanhTanhsimple_rnn_cell_173/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_173_matmul_readvariableop_resource3simple_rnn_cell_173_biasadd_readvariableop_resource4simple_rnn_cell_173_matmul_1_readvariableop_resource*
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
while_body_11269980*
condR
while_cond_11269979*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_173/BiasAdd/ReadVariableOp*^simple_rnn_cell_173/MatMul/ReadVariableOp,^simple_rnn_cell_173/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_173/BiasAdd/ReadVariableOp*simple_rnn_cell_173/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_173/MatMul/ReadVariableOp)simple_rnn_cell_173/MatMul/ReadVariableOp2Z
+simple_rnn_cell_173/MatMul_1/ReadVariableOp+simple_rnn_cell_173/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
while_cond_11269368
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11269368___redundant_placeholder06
2while_while_cond_11269368___redundant_placeholder16
2while_while_cond_11269368___redundant_placeholder26
2while_while_cond_11269368___redundant_placeholder3
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
while_body_11269872
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_173_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_173_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_173_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_173_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_173/MatMul/ReadVariableOp?1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_173_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_173/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_173/BiasAddBiasAdd*while/simple_rnn_cell_173/MatMul:product:08while/simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_173/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_173/addAddV2*while/simple_rnn_cell_173/BiasAdd:output:0,while/simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_173/TanhTanh!while/simple_rnn_cell_173/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_173/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_173/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_173/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_173/MatMul/ReadVariableOp2^while/simple_rnn_cell_173/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_173_biasadd_readvariableop_resource;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_173_matmul_1_readvariableop_resource<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_173_matmul_readvariableop_resource:while_simple_rnn_cell_173_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_173/MatMul/ReadVariableOp/while/simple_rnn_cell_173/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp: 
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
?!
?
while_body_11267278
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_172_11267300_0:P2
$while_simple_rnn_cell_172_11267302_0:P6
$while_simple_rnn_cell_172_11267304_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_172_11267300:P0
"while_simple_rnn_cell_172_11267302:P4
"while_simple_rnn_cell_172_11267304:PP??1while/simple_rnn_cell_172/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/simple_rnn_cell_172/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_172_11267300_0$while_simple_rnn_cell_172_11267302_0$while_simple_rnn_cell_172_11267304_0*
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
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11267265?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_172/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_172/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp2^while/simple_rnn_cell_172/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_172_11267300$while_simple_rnn_cell_172_11267300_0"J
"while_simple_rnn_cell_172_11267302$while_simple_rnn_cell_172_11267302_0"J
"while_simple_rnn_cell_172_11267304$while_simple_rnn_cell_172_11267304_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2f
1while/simple_rnn_cell_172/StatefulPartitionedCall1while/simple_rnn_cell_172/StatefulPartitionedCall: 
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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11269830
inputs_0D
2simple_rnn_cell_173_matmul_readvariableop_resource:PdA
3simple_rnn_cell_173_biasadd_readvariableop_resource:dF
4simple_rnn_cell_173_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_173/BiasAdd/ReadVariableOp?)simple_rnn_cell_173/MatMul/ReadVariableOp?+simple_rnn_cell_173/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_173_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_173/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_173_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_173/BiasAddBiasAdd$simple_rnn_cell_173/MatMul:product:02simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_173_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_173/MatMul_1MatMulzeros:output:03simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_173/addAddV2$simple_rnn_cell_173/BiasAdd:output:0&simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_173/TanhTanhsimple_rnn_cell_173/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_173_matmul_readvariableop_resource3simple_rnn_cell_173_biasadd_readvariableop_resource4simple_rnn_cell_173_matmul_1_readvariableop_resource*
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
while_body_11269764*
condR
while_cond_11269763*8
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
NoOpNoOp+^simple_rnn_cell_173/BiasAdd/ReadVariableOp*^simple_rnn_cell_173/MatMul/ReadVariableOp,^simple_rnn_cell_173/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2X
*simple_rnn_cell_173/BiasAdd/ReadVariableOp*simple_rnn_cell_173/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_173/MatMul/ReadVariableOp)simple_rnn_cell_173/MatMul/ReadVariableOp2Z
+simple_rnn_cell_173/MatMul_1/ReadVariableOp+simple_rnn_cell_173/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?
?
*__inference_dense_6_layer_call_fn_11270082

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
E__inference_dense_6_layer_call_and_return_conditional_losses_11268062o
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
?4
?
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11267792

inputs.
simple_rnn_cell_173_11267717:Pd*
simple_rnn_cell_173_11267719:d.
simple_rnn_cell_173_11267721:dd
identity??+simple_rnn_cell_173/StatefulPartitionedCall?while;
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
+simple_rnn_cell_173/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_173_11267717simple_rnn_cell_173_11267719simple_rnn_cell_173_11267721*
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
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11267677n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_173_11267717simple_rnn_cell_173_11267719simple_rnn_cell_173_11267721*
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
while_body_11267729*
condR
while_cond_11267728*8
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
NoOpNoOp,^simple_rnn_cell_173/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2Z
+simple_rnn_cell_173/StatefulPartitionedCall+simple_rnn_cell_173/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?-
?
while_body_11269153
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_172_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_172_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_172_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_172_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_172/MatMul/ReadVariableOp?1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_172_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_172/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_172/BiasAddBiasAdd*while/simple_rnn_cell_172/MatMul:product:08while/simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_172/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_172/addAddV2*while/simple_rnn_cell_172/BiasAdd:output:0,while/simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_172/TanhTanh!while/simple_rnn_cell_172/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_172/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_172/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_172/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_172/MatMul/ReadVariableOp2^while/simple_rnn_cell_172/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_172_biasadd_readvariableop_resource;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_172_matmul_1_readvariableop_resource<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_172_matmul_readvariableop_resource:while_simple_rnn_cell_172_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_172/MatMul/ReadVariableOp/while/simple_rnn_cell_172/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp: 
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
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268542
simple_rnn_12_input(
simple_rnn_12_11268520:P$
simple_rnn_12_11268522:P(
simple_rnn_12_11268524:PP(
simple_rnn_13_11268528:Pd$
simple_rnn_13_11268530:d(
simple_rnn_13_11268532:dd"
dense_6_11268536:d
dense_6_11268538:
identity??dense_6/StatefulPartitionedCall?"dropout_12/StatefulPartitionedCall?"dropout_13/StatefulPartitionedCall?%simple_rnn_12/StatefulPartitionedCall?%simple_rnn_13/StatefulPartitionedCall?
%simple_rnn_12/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_12_inputsimple_rnn_12_11268520simple_rnn_12_11268522simple_rnn_12_11268524*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11268395?
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_11268271?
%simple_rnn_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0simple_rnn_13_11268528simple_rnn_13_11268530simple_rnn_13_11268532*
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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11268242?
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_13/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
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
H__inference_dropout_13_layer_call_and_return_conditional_losses_11268118?
dense_6/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_6_11268536dense_6_11268538*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_11268062w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_6/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall&^simple_rnn_12/StatefulPartitionedCall&^simple_rnn_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2N
%simple_rnn_12/StatefulPartitionedCall%simple_rnn_12/StatefulPartitionedCall2N
%simple_rnn_13/StatefulPartitionedCall%simple_rnn_13/StatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_12_input
?
f
H__inference_dropout_13_layer_call_and_return_conditional_losses_11268050

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
?
?
while_cond_11267277
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11267277___redundant_placeholder06
2while_while_cond_11267277___redundant_placeholder16
2while_while_cond_11267277___redundant_placeholder26
2while_while_cond_11267277___redundant_placeholder3
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
while_cond_11268175
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11268175___redundant_placeholder06
2while_while_cond_11268175___redundant_placeholder16
2while_while_cond_11268175___redundant_placeholder26
2while_while_cond_11268175___redundant_placeholder3
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
while_cond_11269871
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11269871___redundant_placeholder06
2while_while_cond_11269871___redundant_placeholder16
2while_while_cond_11269871___redundant_placeholder26
2while_while_cond_11269871___redundant_placeholder3
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
0__inference_simple_rnn_12_layer_call_fn_11269089
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
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11267500|
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

?
6__inference_simple_rnn_cell_173_layer_call_fn_11270168

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
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11267557o
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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11268037

inputsD
2simple_rnn_cell_173_matmul_readvariableop_resource:PdA
3simple_rnn_cell_173_biasadd_readvariableop_resource:dF
4simple_rnn_cell_173_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_173/BiasAdd/ReadVariableOp?)simple_rnn_cell_173/MatMul/ReadVariableOp?+simple_rnn_cell_173/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_173_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_173/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_173_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_173/BiasAddBiasAdd$simple_rnn_cell_173/MatMul:product:02simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_173_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_173/MatMul_1MatMulzeros:output:03simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_173/addAddV2$simple_rnn_cell_173/BiasAdd:output:0&simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_173/TanhTanhsimple_rnn_cell_173/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_173_matmul_readvariableop_resource3simple_rnn_cell_173_biasadd_readvariableop_resource4simple_rnn_cell_173_matmul_1_readvariableop_resource*
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
while_body_11267971*
condR
while_cond_11267970*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_173/BiasAdd/ReadVariableOp*^simple_rnn_cell_173/MatMul/ReadVariableOp,^simple_rnn_cell_173/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_173/BiasAdd/ReadVariableOp*simple_rnn_cell_173/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_173/MatMul/ReadVariableOp)simple_rnn_cell_173/MatMul/ReadVariableOp2Z
+simple_rnn_cell_173/MatMul_1/ReadVariableOp+simple_rnn_cell_173/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11270199

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
??
?	
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268810

inputsR
@simple_rnn_12_simple_rnn_cell_172_matmul_readvariableop_resource:PO
Asimple_rnn_12_simple_rnn_cell_172_biasadd_readvariableop_resource:PT
Bsimple_rnn_12_simple_rnn_cell_172_matmul_1_readvariableop_resource:PPR
@simple_rnn_13_simple_rnn_cell_173_matmul_readvariableop_resource:PdO
Asimple_rnn_13_simple_rnn_cell_173_biasadd_readvariableop_resource:dT
Bsimple_rnn_13_simple_rnn_cell_173_matmul_1_readvariableop_resource:dd8
&dense_6_matmul_readvariableop_resource:d5
'dense_6_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?8simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOp?7simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOp?9simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp?simple_rnn_12/while?8simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOp?7simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOp?9simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp?simple_rnn_13/whileI
simple_rnn_12/ShapeShapeinputs*
T0*
_output_shapes
:k
!simple_rnn_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_12/strided_sliceStridedSlicesimple_rnn_12/Shape:output:0*simple_rnn_12/strided_slice/stack:output:0,simple_rnn_12/strided_slice/stack_1:output:0,simple_rnn_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
simple_rnn_12/zeros/packedPack$simple_rnn_12/strided_slice:output:0%simple_rnn_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_12/zerosFill#simple_rnn_12/zeros/packed:output:0"simple_rnn_12/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pq
simple_rnn_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_12/transpose	Transposeinputs%simple_rnn_12/transpose/perm:output:0*
T0*+
_output_shapes
:?????????`
simple_rnn_12/Shape_1Shapesimple_rnn_12/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_12/strided_slice_1StridedSlicesimple_rnn_12/Shape_1:output:0,simple_rnn_12/strided_slice_1/stack:output:0.simple_rnn_12/strided_slice_1/stack_1:output:0.simple_rnn_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_12/TensorArrayV2TensorListReserve2simple_rnn_12/TensorArrayV2/element_shape:output:0&simple_rnn_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Csimple_rnn_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
5simple_rnn_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_12/transpose:y:0Lsimple_rnn_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???m
#simple_rnn_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_12/strided_slice_2StridedSlicesimple_rnn_12/transpose:y:0,simple_rnn_12/strided_slice_2/stack:output:0.simple_rnn_12/strided_slice_2/stack_1:output:0.simple_rnn_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
7simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp@simple_rnn_12_simple_rnn_cell_172_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
(simple_rnn_12/simple_rnn_cell_172/MatMulMatMul&simple_rnn_12/strided_slice_2:output:0?simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOpAsimple_rnn_12_simple_rnn_cell_172_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
)simple_rnn_12/simple_rnn_cell_172/BiasAddBiasAdd2simple_rnn_12/simple_rnn_cell_172/MatMul:product:0@simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
9simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOpBsimple_rnn_12_simple_rnn_cell_172_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
*simple_rnn_12/simple_rnn_cell_172/MatMul_1MatMulsimple_rnn_12/zeros:output:0Asimple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
%simple_rnn_12/simple_rnn_cell_172/addAddV22simple_rnn_12/simple_rnn_cell_172/BiasAdd:output:04simple_rnn_12/simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
&simple_rnn_12/simple_rnn_cell_172/TanhTanh)simple_rnn_12/simple_rnn_cell_172/add:z:0*
T0*'
_output_shapes
:?????????P|
+simple_rnn_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
simple_rnn_12/TensorArrayV2_1TensorListReserve4simple_rnn_12/TensorArrayV2_1/element_shape:output:0&simple_rnn_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???T
simple_rnn_12/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 simple_rnn_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_12/whileWhile)simple_rnn_12/while/loop_counter:output:0/simple_rnn_12/while/maximum_iterations:output:0simple_rnn_12/time:output:0&simple_rnn_12/TensorArrayV2_1:handle:0simple_rnn_12/zeros:output:0&simple_rnn_12/strided_slice_1:output:0Esimple_rnn_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0@simple_rnn_12_simple_rnn_cell_172_matmul_readvariableop_resourceAsimple_rnn_12_simple_rnn_cell_172_biasadd_readvariableop_resourceBsimple_rnn_12_simple_rnn_cell_172_matmul_1_readvariableop_resource*
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
!simple_rnn_12_while_body_11268632*-
cond%R#
!simple_rnn_12_while_cond_11268631*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
>simple_rnn_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
0simple_rnn_12/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_12/while:output:3Gsimple_rnn_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0v
#simple_rnn_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%simple_rnn_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_12/strided_slice_3StridedSlice9simple_rnn_12/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_12/strided_slice_3/stack:output:0.simple_rnn_12/strided_slice_3/stack_1:output:0.simple_rnn_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_masks
simple_rnn_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_12/transpose_1	Transpose9simple_rnn_12/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pt
dropout_12/IdentityIdentitysimple_rnn_12/transpose_1:y:0*
T0*+
_output_shapes
:?????????P_
simple_rnn_13/ShapeShapedropout_12/Identity:output:0*
T0*
_output_shapes
:k
!simple_rnn_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_13/strided_sliceStridedSlicesimple_rnn_13/Shape:output:0*simple_rnn_13/strided_slice/stack:output:0,simple_rnn_13/strided_slice/stack_1:output:0,simple_rnn_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
simple_rnn_13/zeros/packedPack$simple_rnn_13/strided_slice:output:0%simple_rnn_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_13/zerosFill#simple_rnn_13/zeros/packed:output:0"simple_rnn_13/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dq
simple_rnn_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_13/transpose	Transposedropout_12/Identity:output:0%simple_rnn_13/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P`
simple_rnn_13/Shape_1Shapesimple_rnn_13/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_13/strided_slice_1StridedSlicesimple_rnn_13/Shape_1:output:0,simple_rnn_13/strided_slice_1/stack:output:0.simple_rnn_13/strided_slice_1/stack_1:output:0.simple_rnn_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_13/TensorArrayV2TensorListReserve2simple_rnn_13/TensorArrayV2/element_shape:output:0&simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Csimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
5simple_rnn_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_13/transpose:y:0Lsimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???m
#simple_rnn_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_13/strided_slice_2StridedSlicesimple_rnn_13/transpose:y:0,simple_rnn_13/strided_slice_2/stack:output:0.simple_rnn_13/strided_slice_2/stack_1:output:0.simple_rnn_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
7simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp@simple_rnn_13_simple_rnn_cell_173_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
(simple_rnn_13/simple_rnn_cell_173/MatMulMatMul&simple_rnn_13/strided_slice_2:output:0?simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOpAsimple_rnn_13_simple_rnn_cell_173_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
)simple_rnn_13/simple_rnn_cell_173/BiasAddBiasAdd2simple_rnn_13/simple_rnn_cell_173/MatMul:product:0@simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
9simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOpBsimple_rnn_13_simple_rnn_cell_173_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
*simple_rnn_13/simple_rnn_cell_173/MatMul_1MatMulsimple_rnn_13/zeros:output:0Asimple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
%simple_rnn_13/simple_rnn_cell_173/addAddV22simple_rnn_13/simple_rnn_cell_173/BiasAdd:output:04simple_rnn_13/simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
&simple_rnn_13/simple_rnn_cell_173/TanhTanh)simple_rnn_13/simple_rnn_cell_173/add:z:0*
T0*'
_output_shapes
:?????????d|
+simple_rnn_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
simple_rnn_13/TensorArrayV2_1TensorListReserve4simple_rnn_13/TensorArrayV2_1/element_shape:output:0&simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???T
simple_rnn_13/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 simple_rnn_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_13/whileWhile)simple_rnn_13/while/loop_counter:output:0/simple_rnn_13/while/maximum_iterations:output:0simple_rnn_13/time:output:0&simple_rnn_13/TensorArrayV2_1:handle:0simple_rnn_13/zeros:output:0&simple_rnn_13/strided_slice_1:output:0Esimple_rnn_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0@simple_rnn_13_simple_rnn_cell_173_matmul_readvariableop_resourceAsimple_rnn_13_simple_rnn_cell_173_biasadd_readvariableop_resourceBsimple_rnn_13_simple_rnn_cell_173_matmul_1_readvariableop_resource*
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
!simple_rnn_13_while_body_11268737*-
cond%R#
!simple_rnn_13_while_cond_11268736*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
>simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
0simple_rnn_13/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_13/while:output:3Gsimple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0v
#simple_rnn_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%simple_rnn_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_13/strided_slice_3StridedSlice9simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_13/strided_slice_3/stack:output:0.simple_rnn_13/strided_slice_3/stack_1:output:0.simple_rnn_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_masks
simple_rnn_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_13/transpose_1	Transpose9simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_13/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dy
dropout_13/IdentityIdentity&simple_rnn_13/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_6/MatMulMatMuldropout_13/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp9^simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOp8^simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOp:^simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp^simple_rnn_12/while9^simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOp8^simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOp:^simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp^simple_rnn_13/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2t
8simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOp8simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOp2r
7simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOp7simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOp2v
9simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp9simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp2*
simple_rnn_12/whilesimple_rnn_12/while2t
8simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOp8simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOp2r
7simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOp7simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOp2v
9simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp9simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp2*
simple_rnn_13/whilesimple_rnn_13/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?>
?
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11269722
inputs_0D
2simple_rnn_cell_173_matmul_readvariableop_resource:PdA
3simple_rnn_cell_173_biasadd_readvariableop_resource:dF
4simple_rnn_cell_173_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_173/BiasAdd/ReadVariableOp?)simple_rnn_cell_173/MatMul/ReadVariableOp?+simple_rnn_cell_173/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_173_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_173/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_173_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_173/BiasAddBiasAdd$simple_rnn_cell_173/MatMul:product:02simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_173_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_173/MatMul_1MatMulzeros:output:03simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_173/addAddV2$simple_rnn_cell_173/BiasAdd:output:0&simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_173/TanhTanhsimple_rnn_cell_173/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_173_matmul_readvariableop_resource3simple_rnn_cell_173_biasadd_readvariableop_resource4simple_rnn_cell_173_matmul_1_readvariableop_resource*
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
while_body_11269656*
condR
while_cond_11269655*8
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
NoOpNoOp+^simple_rnn_cell_173/BiasAdd/ReadVariableOp*^simple_rnn_cell_173/MatMul/ReadVariableOp,^simple_rnn_cell_173/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2X
*simple_rnn_cell_173/BiasAdd/ReadVariableOp*simple_rnn_cell_173/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_173/MatMul/ReadVariableOp)simple_rnn_cell_173/MatMul/ReadVariableOp2Z
+simple_rnn_cell_173/MatMul_1/ReadVariableOp+simple_rnn_cell_173/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
̫
?	
J__inference_sequential_6_layer_call_and_return_conditional_losses_11269044

inputsR
@simple_rnn_12_simple_rnn_cell_172_matmul_readvariableop_resource:PO
Asimple_rnn_12_simple_rnn_cell_172_biasadd_readvariableop_resource:PT
Bsimple_rnn_12_simple_rnn_cell_172_matmul_1_readvariableop_resource:PPR
@simple_rnn_13_simple_rnn_cell_173_matmul_readvariableop_resource:PdO
Asimple_rnn_13_simple_rnn_cell_173_biasadd_readvariableop_resource:dT
Bsimple_rnn_13_simple_rnn_cell_173_matmul_1_readvariableop_resource:dd8
&dense_6_matmul_readvariableop_resource:d5
'dense_6_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?8simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOp?7simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOp?9simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp?simple_rnn_12/while?8simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOp?7simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOp?9simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp?simple_rnn_13/whileI
simple_rnn_12/ShapeShapeinputs*
T0*
_output_shapes
:k
!simple_rnn_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_12/strided_sliceStridedSlicesimple_rnn_12/Shape:output:0*simple_rnn_12/strided_slice/stack:output:0,simple_rnn_12/strided_slice/stack_1:output:0,simple_rnn_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
simple_rnn_12/zeros/packedPack$simple_rnn_12/strided_slice:output:0%simple_rnn_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_12/zerosFill#simple_rnn_12/zeros/packed:output:0"simple_rnn_12/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pq
simple_rnn_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_12/transpose	Transposeinputs%simple_rnn_12/transpose/perm:output:0*
T0*+
_output_shapes
:?????????`
simple_rnn_12/Shape_1Shapesimple_rnn_12/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_12/strided_slice_1StridedSlicesimple_rnn_12/Shape_1:output:0,simple_rnn_12/strided_slice_1/stack:output:0.simple_rnn_12/strided_slice_1/stack_1:output:0.simple_rnn_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_12/TensorArrayV2TensorListReserve2simple_rnn_12/TensorArrayV2/element_shape:output:0&simple_rnn_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Csimple_rnn_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
5simple_rnn_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_12/transpose:y:0Lsimple_rnn_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???m
#simple_rnn_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_12/strided_slice_2StridedSlicesimple_rnn_12/transpose:y:0,simple_rnn_12/strided_slice_2/stack:output:0.simple_rnn_12/strided_slice_2/stack_1:output:0.simple_rnn_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
7simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp@simple_rnn_12_simple_rnn_cell_172_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
(simple_rnn_12/simple_rnn_cell_172/MatMulMatMul&simple_rnn_12/strided_slice_2:output:0?simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOpAsimple_rnn_12_simple_rnn_cell_172_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
)simple_rnn_12/simple_rnn_cell_172/BiasAddBiasAdd2simple_rnn_12/simple_rnn_cell_172/MatMul:product:0@simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
9simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOpBsimple_rnn_12_simple_rnn_cell_172_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
*simple_rnn_12/simple_rnn_cell_172/MatMul_1MatMulsimple_rnn_12/zeros:output:0Asimple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
%simple_rnn_12/simple_rnn_cell_172/addAddV22simple_rnn_12/simple_rnn_cell_172/BiasAdd:output:04simple_rnn_12/simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
&simple_rnn_12/simple_rnn_cell_172/TanhTanh)simple_rnn_12/simple_rnn_cell_172/add:z:0*
T0*'
_output_shapes
:?????????P|
+simple_rnn_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
simple_rnn_12/TensorArrayV2_1TensorListReserve4simple_rnn_12/TensorArrayV2_1/element_shape:output:0&simple_rnn_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???T
simple_rnn_12/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 simple_rnn_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_12/whileWhile)simple_rnn_12/while/loop_counter:output:0/simple_rnn_12/while/maximum_iterations:output:0simple_rnn_12/time:output:0&simple_rnn_12/TensorArrayV2_1:handle:0simple_rnn_12/zeros:output:0&simple_rnn_12/strided_slice_1:output:0Esimple_rnn_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0@simple_rnn_12_simple_rnn_cell_172_matmul_readvariableop_resourceAsimple_rnn_12_simple_rnn_cell_172_biasadd_readvariableop_resourceBsimple_rnn_12_simple_rnn_cell_172_matmul_1_readvariableop_resource*
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
!simple_rnn_12_while_body_11268852*-
cond%R#
!simple_rnn_12_while_cond_11268851*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
>simple_rnn_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
0simple_rnn_12/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_12/while:output:3Gsimple_rnn_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0v
#simple_rnn_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%simple_rnn_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_12/strided_slice_3StridedSlice9simple_rnn_12/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_12/strided_slice_3/stack:output:0.simple_rnn_12/strided_slice_3/stack_1:output:0.simple_rnn_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_masks
simple_rnn_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_12/transpose_1	Transpose9simple_rnn_12/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P]
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_12/dropout/MulMulsimple_rnn_12/transpose_1:y:0!dropout_12/dropout/Const:output:0*
T0*+
_output_shapes
:?????????Pe
dropout_12/dropout/ShapeShapesimple_rnn_12/transpose_1:y:0*
T0*
_output_shapes
:?
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
dtype0f
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????P?
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????P?
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P_
simple_rnn_13/ShapeShapedropout_12/dropout/Mul_1:z:0*
T0*
_output_shapes
:k
!simple_rnn_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_13/strided_sliceStridedSlicesimple_rnn_13/Shape:output:0*simple_rnn_13/strided_slice/stack:output:0,simple_rnn_13/strided_slice/stack_1:output:0,simple_rnn_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
simple_rnn_13/zeros/packedPack$simple_rnn_13/strided_slice:output:0%simple_rnn_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_13/zerosFill#simple_rnn_13/zeros/packed:output:0"simple_rnn_13/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dq
simple_rnn_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_13/transpose	Transposedropout_12/dropout/Mul_1:z:0%simple_rnn_13/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P`
simple_rnn_13/Shape_1Shapesimple_rnn_13/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_13/strided_slice_1StridedSlicesimple_rnn_13/Shape_1:output:0,simple_rnn_13/strided_slice_1/stack:output:0.simple_rnn_13/strided_slice_1/stack_1:output:0.simple_rnn_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_13/TensorArrayV2TensorListReserve2simple_rnn_13/TensorArrayV2/element_shape:output:0&simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Csimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
5simple_rnn_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_13/transpose:y:0Lsimple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???m
#simple_rnn_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_13/strided_slice_2StridedSlicesimple_rnn_13/transpose:y:0,simple_rnn_13/strided_slice_2/stack:output:0.simple_rnn_13/strided_slice_2/stack_1:output:0.simple_rnn_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
7simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp@simple_rnn_13_simple_rnn_cell_173_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
(simple_rnn_13/simple_rnn_cell_173/MatMulMatMul&simple_rnn_13/strided_slice_2:output:0?simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOpAsimple_rnn_13_simple_rnn_cell_173_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
)simple_rnn_13/simple_rnn_cell_173/BiasAddBiasAdd2simple_rnn_13/simple_rnn_cell_173/MatMul:product:0@simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
9simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOpBsimple_rnn_13_simple_rnn_cell_173_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
*simple_rnn_13/simple_rnn_cell_173/MatMul_1MatMulsimple_rnn_13/zeros:output:0Asimple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
%simple_rnn_13/simple_rnn_cell_173/addAddV22simple_rnn_13/simple_rnn_cell_173/BiasAdd:output:04simple_rnn_13/simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
&simple_rnn_13/simple_rnn_cell_173/TanhTanh)simple_rnn_13/simple_rnn_cell_173/add:z:0*
T0*'
_output_shapes
:?????????d|
+simple_rnn_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
simple_rnn_13/TensorArrayV2_1TensorListReserve4simple_rnn_13/TensorArrayV2_1/element_shape:output:0&simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???T
simple_rnn_13/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 simple_rnn_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_13/whileWhile)simple_rnn_13/while/loop_counter:output:0/simple_rnn_13/while/maximum_iterations:output:0simple_rnn_13/time:output:0&simple_rnn_13/TensorArrayV2_1:handle:0simple_rnn_13/zeros:output:0&simple_rnn_13/strided_slice_1:output:0Esimple_rnn_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0@simple_rnn_13_simple_rnn_cell_173_matmul_readvariableop_resourceAsimple_rnn_13_simple_rnn_cell_173_biasadd_readvariableop_resourceBsimple_rnn_13_simple_rnn_cell_173_matmul_1_readvariableop_resource*
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
!simple_rnn_13_while_body_11268964*-
cond%R#
!simple_rnn_13_while_cond_11268963*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
>simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
0simple_rnn_13/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_13/while:output:3Gsimple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0v
#simple_rnn_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%simple_rnn_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_13/strided_slice_3StridedSlice9simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_13/strided_slice_3/stack:output:0.simple_rnn_13/strided_slice_3/stack_1:output:0.simple_rnn_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_masks
simple_rnn_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_13/transpose_1	Transpose9simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_13/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d]
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_13/dropout/MulMul&simple_rnn_13/strided_slice_3:output:0!dropout_13/dropout/Const:output:0*
T0*'
_output_shapes
:?????????dn
dropout_13/dropout/ShapeShape&simple_rnn_13/strided_slice_3:output:0*
T0*
_output_shapes
:?
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0f
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_6/MatMulMatMuldropout_13/dropout/Mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp9^simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOp8^simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOp:^simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp^simple_rnn_12/while9^simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOp8^simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOp:^simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp^simple_rnn_13/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2t
8simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOp8simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOp2r
7simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOp7simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOp2v
9simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp9simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp2*
simple_rnn_12/whilesimple_rnn_12/while2t
8simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOp8simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOp2r
7simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOp7simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOp2v
9simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp9simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp2*
simple_rnn_13/whilesimple_rnn_13/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ѷ
?

#__inference__wrapped_model_11267217
simple_rnn_12_input_
Msequential_6_simple_rnn_12_simple_rnn_cell_172_matmul_readvariableop_resource:P\
Nsequential_6_simple_rnn_12_simple_rnn_cell_172_biasadd_readvariableop_resource:Pa
Osequential_6_simple_rnn_12_simple_rnn_cell_172_matmul_1_readvariableop_resource:PP_
Msequential_6_simple_rnn_13_simple_rnn_cell_173_matmul_readvariableop_resource:Pd\
Nsequential_6_simple_rnn_13_simple_rnn_cell_173_biasadd_readvariableop_resource:da
Osequential_6_simple_rnn_13_simple_rnn_cell_173_matmul_1_readvariableop_resource:ddE
3sequential_6_dense_6_matmul_readvariableop_resource:dB
4sequential_6_dense_6_biasadd_readvariableop_resource:
identity??+sequential_6/dense_6/BiasAdd/ReadVariableOp?*sequential_6/dense_6/MatMul/ReadVariableOp?Esequential_6/simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOp?Dsequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOp?Fsequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp? sequential_6/simple_rnn_12/while?Esequential_6/simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOp?Dsequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOp?Fsequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp? sequential_6/simple_rnn_13/whilec
 sequential_6/simple_rnn_12/ShapeShapesimple_rnn_12_input*
T0*
_output_shapes
:x
.sequential_6/simple_rnn_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_6/simple_rnn_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_6/simple_rnn_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential_6/simple_rnn_12/strided_sliceStridedSlice)sequential_6/simple_rnn_12/Shape:output:07sequential_6/simple_rnn_12/strided_slice/stack:output:09sequential_6/simple_rnn_12/strided_slice/stack_1:output:09sequential_6/simple_rnn_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_6/simple_rnn_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
'sequential_6/simple_rnn_12/zeros/packedPack1sequential_6/simple_rnn_12/strided_slice:output:02sequential_6/simple_rnn_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&sequential_6/simple_rnn_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 sequential_6/simple_rnn_12/zerosFill0sequential_6/simple_rnn_12/zeros/packed:output:0/sequential_6/simple_rnn_12/zeros/Const:output:0*
T0*'
_output_shapes
:?????????P~
)sequential_6/simple_rnn_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
$sequential_6/simple_rnn_12/transpose	Transposesimple_rnn_12_input2sequential_6/simple_rnn_12/transpose/perm:output:0*
T0*+
_output_shapes
:?????????z
"sequential_6/simple_rnn_12/Shape_1Shape(sequential_6/simple_rnn_12/transpose:y:0*
T0*
_output_shapes
:z
0sequential_6/simple_rnn_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_6/simple_rnn_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_6/simple_rnn_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_6/simple_rnn_12/strided_slice_1StridedSlice+sequential_6/simple_rnn_12/Shape_1:output:09sequential_6/simple_rnn_12/strided_slice_1/stack:output:0;sequential_6/simple_rnn_12/strided_slice_1/stack_1:output:0;sequential_6/simple_rnn_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
6sequential_6/simple_rnn_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
(sequential_6/simple_rnn_12/TensorArrayV2TensorListReserve?sequential_6/simple_rnn_12/TensorArrayV2/element_shape:output:03sequential_6/simple_rnn_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Psequential_6/simple_rnn_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Bsequential_6/simple_rnn_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_6/simple_rnn_12/transpose:y:0Ysequential_6/simple_rnn_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???z
0sequential_6/simple_rnn_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_6/simple_rnn_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_6/simple_rnn_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_6/simple_rnn_12/strided_slice_2StridedSlice(sequential_6/simple_rnn_12/transpose:y:09sequential_6/simple_rnn_12/strided_slice_2/stack:output:0;sequential_6/simple_rnn_12/strided_slice_2/stack_1:output:0;sequential_6/simple_rnn_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
Dsequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOpMsequential_6_simple_rnn_12_simple_rnn_cell_172_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
5sequential_6/simple_rnn_12/simple_rnn_cell_172/MatMulMatMul3sequential_6/simple_rnn_12/strided_slice_2:output:0Lsequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Esequential_6/simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOpNsequential_6_simple_rnn_12_simple_rnn_cell_172_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
6sequential_6/simple_rnn_12/simple_rnn_cell_172/BiasAddBiasAdd?sequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul:product:0Msequential_6/simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Fsequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOpOsequential_6_simple_rnn_12_simple_rnn_cell_172_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
7sequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul_1MatMul)sequential_6/simple_rnn_12/zeros:output:0Nsequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
2sequential_6/simple_rnn_12/simple_rnn_cell_172/addAddV2?sequential_6/simple_rnn_12/simple_rnn_cell_172/BiasAdd:output:0Asequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
3sequential_6/simple_rnn_12/simple_rnn_cell_172/TanhTanh6sequential_6/simple_rnn_12/simple_rnn_cell_172/add:z:0*
T0*'
_output_shapes
:?????????P?
8sequential_6/simple_rnn_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
*sequential_6/simple_rnn_12/TensorArrayV2_1TensorListReserveAsequential_6/simple_rnn_12/TensorArrayV2_1/element_shape:output:03sequential_6/simple_rnn_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???a
sequential_6/simple_rnn_12/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_6/simple_rnn_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
-sequential_6/simple_rnn_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
 sequential_6/simple_rnn_12/whileWhile6sequential_6/simple_rnn_12/while/loop_counter:output:0<sequential_6/simple_rnn_12/while/maximum_iterations:output:0(sequential_6/simple_rnn_12/time:output:03sequential_6/simple_rnn_12/TensorArrayV2_1:handle:0)sequential_6/simple_rnn_12/zeros:output:03sequential_6/simple_rnn_12/strided_slice_1:output:0Rsequential_6/simple_rnn_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_6_simple_rnn_12_simple_rnn_cell_172_matmul_readvariableop_resourceNsequential_6_simple_rnn_12_simple_rnn_cell_172_biasadd_readvariableop_resourceOsequential_6_simple_rnn_12_simple_rnn_cell_172_matmul_1_readvariableop_resource*
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
.sequential_6_simple_rnn_12_while_body_11267039*:
cond2R0
.sequential_6_simple_rnn_12_while_cond_11267038*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
Ksequential_6/simple_rnn_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
=sequential_6/simple_rnn_12/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_6/simple_rnn_12/while:output:3Tsequential_6/simple_rnn_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0?
0sequential_6/simple_rnn_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????|
2sequential_6/simple_rnn_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_6/simple_rnn_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_6/simple_rnn_12/strided_slice_3StridedSliceFsequential_6/simple_rnn_12/TensorArrayV2Stack/TensorListStack:tensor:09sequential_6/simple_rnn_12/strided_slice_3/stack:output:0;sequential_6/simple_rnn_12/strided_slice_3/stack_1:output:0;sequential_6/simple_rnn_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
+sequential_6/simple_rnn_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
&sequential_6/simple_rnn_12/transpose_1	TransposeFsequential_6/simple_rnn_12/TensorArrayV2Stack/TensorListStack:tensor:04sequential_6/simple_rnn_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P?
 sequential_6/dropout_12/IdentityIdentity*sequential_6/simple_rnn_12/transpose_1:y:0*
T0*+
_output_shapes
:?????????Py
 sequential_6/simple_rnn_13/ShapeShape)sequential_6/dropout_12/Identity:output:0*
T0*
_output_shapes
:x
.sequential_6/simple_rnn_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_6/simple_rnn_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_6/simple_rnn_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential_6/simple_rnn_13/strided_sliceStridedSlice)sequential_6/simple_rnn_13/Shape:output:07sequential_6/simple_rnn_13/strided_slice/stack:output:09sequential_6/simple_rnn_13/strided_slice/stack_1:output:09sequential_6/simple_rnn_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_6/simple_rnn_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
'sequential_6/simple_rnn_13/zeros/packedPack1sequential_6/simple_rnn_13/strided_slice:output:02sequential_6/simple_rnn_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&sequential_6/simple_rnn_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 sequential_6/simple_rnn_13/zerosFill0sequential_6/simple_rnn_13/zeros/packed:output:0/sequential_6/simple_rnn_13/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d~
)sequential_6/simple_rnn_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
$sequential_6/simple_rnn_13/transpose	Transpose)sequential_6/dropout_12/Identity:output:02sequential_6/simple_rnn_13/transpose/perm:output:0*
T0*+
_output_shapes
:?????????Pz
"sequential_6/simple_rnn_13/Shape_1Shape(sequential_6/simple_rnn_13/transpose:y:0*
T0*
_output_shapes
:z
0sequential_6/simple_rnn_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_6/simple_rnn_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_6/simple_rnn_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_6/simple_rnn_13/strided_slice_1StridedSlice+sequential_6/simple_rnn_13/Shape_1:output:09sequential_6/simple_rnn_13/strided_slice_1/stack:output:0;sequential_6/simple_rnn_13/strided_slice_1/stack_1:output:0;sequential_6/simple_rnn_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
6sequential_6/simple_rnn_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
(sequential_6/simple_rnn_13/TensorArrayV2TensorListReserve?sequential_6/simple_rnn_13/TensorArrayV2/element_shape:output:03sequential_6/simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Psequential_6/simple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
Bsequential_6/simple_rnn_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_6/simple_rnn_13/transpose:y:0Ysequential_6/simple_rnn_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???z
0sequential_6/simple_rnn_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_6/simple_rnn_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_6/simple_rnn_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_6/simple_rnn_13/strided_slice_2StridedSlice(sequential_6/simple_rnn_13/transpose:y:09sequential_6/simple_rnn_13/strided_slice_2/stack:output:0;sequential_6/simple_rnn_13/strided_slice_2/stack_1:output:0;sequential_6/simple_rnn_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
Dsequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOpMsequential_6_simple_rnn_13_simple_rnn_cell_173_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
5sequential_6/simple_rnn_13/simple_rnn_cell_173/MatMulMatMul3sequential_6/simple_rnn_13/strided_slice_2:output:0Lsequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Esequential_6/simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOpNsequential_6_simple_rnn_13_simple_rnn_cell_173_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
6sequential_6/simple_rnn_13/simple_rnn_cell_173/BiasAddBiasAdd?sequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul:product:0Msequential_6/simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Fsequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOpOsequential_6_simple_rnn_13_simple_rnn_cell_173_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
7sequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul_1MatMul)sequential_6/simple_rnn_13/zeros:output:0Nsequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
2sequential_6/simple_rnn_13/simple_rnn_cell_173/addAddV2?sequential_6/simple_rnn_13/simple_rnn_cell_173/BiasAdd:output:0Asequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
3sequential_6/simple_rnn_13/simple_rnn_cell_173/TanhTanh6sequential_6/simple_rnn_13/simple_rnn_cell_173/add:z:0*
T0*'
_output_shapes
:?????????d?
8sequential_6/simple_rnn_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
*sequential_6/simple_rnn_13/TensorArrayV2_1TensorListReserveAsequential_6/simple_rnn_13/TensorArrayV2_1/element_shape:output:03sequential_6/simple_rnn_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???a
sequential_6/simple_rnn_13/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_6/simple_rnn_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
-sequential_6/simple_rnn_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
 sequential_6/simple_rnn_13/whileWhile6sequential_6/simple_rnn_13/while/loop_counter:output:0<sequential_6/simple_rnn_13/while/maximum_iterations:output:0(sequential_6/simple_rnn_13/time:output:03sequential_6/simple_rnn_13/TensorArrayV2_1:handle:0)sequential_6/simple_rnn_13/zeros:output:03sequential_6/simple_rnn_13/strided_slice_1:output:0Rsequential_6/simple_rnn_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_6_simple_rnn_13_simple_rnn_cell_173_matmul_readvariableop_resourceNsequential_6_simple_rnn_13_simple_rnn_cell_173_biasadd_readvariableop_resourceOsequential_6_simple_rnn_13_simple_rnn_cell_173_matmul_1_readvariableop_resource*
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
.sequential_6_simple_rnn_13_while_body_11267144*:
cond2R0
.sequential_6_simple_rnn_13_while_cond_11267143*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
Ksequential_6/simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
=sequential_6/simple_rnn_13/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_6/simple_rnn_13/while:output:3Tsequential_6/simple_rnn_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0?
0sequential_6/simple_rnn_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????|
2sequential_6/simple_rnn_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_6/simple_rnn_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_6/simple_rnn_13/strided_slice_3StridedSliceFsequential_6/simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:09sequential_6/simple_rnn_13/strided_slice_3/stack:output:0;sequential_6/simple_rnn_13/strided_slice_3/stack_1:output:0;sequential_6/simple_rnn_13/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask?
+sequential_6/simple_rnn_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
&sequential_6/simple_rnn_13/transpose_1	TransposeFsequential_6/simple_rnn_13/TensorArrayV2Stack/TensorListStack:tensor:04sequential_6/simple_rnn_13/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d?
 sequential_6/dropout_13/IdentityIdentity3sequential_6/simple_rnn_13/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
*sequential_6/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_6_dense_6_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
sequential_6/dense_6/MatMulMatMul)sequential_6/dropout_13/Identity:output:02sequential_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_6/dense_6/BiasAddBiasAdd%sequential_6/dense_6/MatMul:product:03sequential_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_6/dense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^sequential_6/dense_6/BiasAdd/ReadVariableOp+^sequential_6/dense_6/MatMul/ReadVariableOpF^sequential_6/simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOpE^sequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOpG^sequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp!^sequential_6/simple_rnn_12/whileF^sequential_6/simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOpE^sequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOpG^sequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp!^sequential_6/simple_rnn_13/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2Z
+sequential_6/dense_6/BiasAdd/ReadVariableOp+sequential_6/dense_6/BiasAdd/ReadVariableOp2X
*sequential_6/dense_6/MatMul/ReadVariableOp*sequential_6/dense_6/MatMul/ReadVariableOp2?
Esequential_6/simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOpEsequential_6/simple_rnn_12/simple_rnn_cell_172/BiasAdd/ReadVariableOp2?
Dsequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOpDsequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul/ReadVariableOp2?
Fsequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOpFsequential_6/simple_rnn_12/simple_rnn_cell_172/MatMul_1/ReadVariableOp2D
 sequential_6/simple_rnn_12/while sequential_6/simple_rnn_12/while2?
Esequential_6/simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOpEsequential_6/simple_rnn_13/simple_rnn_cell_173/BiasAdd/ReadVariableOp2?
Dsequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOpDsequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul/ReadVariableOp2?
Fsequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOpFsequential_6/simple_rnn_13/simple_rnn_cell_173/MatMul_1/ReadVariableOp2D
 sequential_6/simple_rnn_13/while sequential_6/simple_rnn_13/while:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_12_input
?
?
while_cond_11268328
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11268328___redundant_placeholder06
2while_while_cond_11268328___redundant_placeholder16
2while_while_cond_11268328___redundant_placeholder26
2while_while_cond_11268328___redundant_placeholder3
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
?I
?
!__inference__traced_save_11270332
file_prefix-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopF
Bsavev2_simple_rnn_12_simple_rnn_cell_12_kernel_read_readvariableopP
Lsavev2_simple_rnn_12_simple_rnn_cell_12_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_12_simple_rnn_cell_12_bias_read_readvariableopF
Bsavev2_simple_rnn_13_simple_rnn_cell_13_kernel_read_readvariableopP
Lsavev2_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_13_simple_rnn_cell_13_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_12_simple_rnn_cell_12_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_12_simple_rnn_cell_12_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_12_simple_rnn_cell_12_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_13_simple_rnn_cell_13_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_13_simple_rnn_cell_13_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_12_simple_rnn_cell_12_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_12_simple_rnn_cell_12_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_12_simple_rnn_cell_12_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_13_simple_rnn_cell_13_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_13_simple_rnn_cell_13_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopBsavev2_simple_rnn_12_simple_rnn_cell_12_kernel_read_readvariableopLsavev2_simple_rnn_12_simple_rnn_cell_12_recurrent_kernel_read_readvariableop@savev2_simple_rnn_12_simple_rnn_cell_12_bias_read_readvariableopBsavev2_simple_rnn_13_simple_rnn_cell_13_kernel_read_readvariableopLsavev2_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_read_readvariableop@savev2_simple_rnn_13_simple_rnn_cell_13_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableopIsavev2_adam_simple_rnn_12_simple_rnn_cell_12_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_12_simple_rnn_cell_12_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_12_simple_rnn_cell_12_bias_m_read_readvariableopIsavev2_adam_simple_rnn_13_simple_rnn_cell_13_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_13_simple_rnn_cell_13_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableopIsavev2_adam_simple_rnn_12_simple_rnn_cell_12_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_12_simple_rnn_cell_12_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_12_simple_rnn_cell_12_bias_v_read_readvariableopIsavev2_adam_simple_rnn_13_simple_rnn_cell_13_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_13_simple_rnn_cell_13_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_13_simple_rnn_cell_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
while_cond_11269260
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11269260___redundant_placeholder06
2while_while_cond_11269260___redundant_placeholder16
2while_while_cond_11269260___redundant_placeholder26
2while_while_cond_11269260___redundant_placeholder3
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
/__inference_sequential_6_layer_call_fn_11268088
simple_rnn_12_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268069o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_12_input
?
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268452

inputs(
simple_rnn_12_11268430:P$
simple_rnn_12_11268432:P(
simple_rnn_12_11268434:PP(
simple_rnn_13_11268438:Pd$
simple_rnn_13_11268440:d(
simple_rnn_13_11268442:dd"
dense_6_11268446:d
dense_6_11268448:
identity??dense_6/StatefulPartitionedCall?"dropout_12/StatefulPartitionedCall?"dropout_13/StatefulPartitionedCall?%simple_rnn_12/StatefulPartitionedCall?%simple_rnn_13/StatefulPartitionedCall?
%simple_rnn_12/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_12_11268430simple_rnn_12_11268432simple_rnn_12_11268434*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11268395?
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_11268271?
%simple_rnn_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0simple_rnn_13_11268438simple_rnn_13_11268440simple_rnn_13_11268442*
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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11268242?
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_13/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
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
H__inference_dropout_13_layer_call_and_return_conditional_losses_11268118?
dense_6/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_6_11268446dense_6_11268448*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_11268062w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_6/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall&^simple_rnn_12/StatefulPartitionedCall&^simple_rnn_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2N
%simple_rnn_12/StatefulPartitionedCall%simple_rnn_12/StatefulPartitionedCall2N
%simple_rnn_13/StatefulPartitionedCall%simple_rnn_13/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
!simple_rnn_13_while_cond_112687368
4simple_rnn_13_while_simple_rnn_13_while_loop_counter>
:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations#
simple_rnn_13_while_placeholder%
!simple_rnn_13_while_placeholder_1%
!simple_rnn_13_while_placeholder_2:
6simple_rnn_13_while_less_simple_rnn_13_strided_slice_1R
Nsimple_rnn_13_while_simple_rnn_13_while_cond_11268736___redundant_placeholder0R
Nsimple_rnn_13_while_simple_rnn_13_while_cond_11268736___redundant_placeholder1R
Nsimple_rnn_13_while_simple_rnn_13_while_cond_11268736___redundant_placeholder2R
Nsimple_rnn_13_while_simple_rnn_13_while_cond_11268736___redundant_placeholder3 
simple_rnn_13_while_identity
?
simple_rnn_13/while/LessLesssimple_rnn_13_while_placeholder6simple_rnn_13_while_less_simple_rnn_13_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_13/while/IdentityIdentitysimple_rnn_13/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_13_while_identity%simple_rnn_13/while/Identity:output:0*(
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
while_cond_11269476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11269476___redundant_placeholder06
2while_while_cond_11269476___redundant_placeholder16
2while_while_cond_11269476___redundant_placeholder26
2while_while_cond_11269476___redundant_placeholder3
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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11269938

inputsD
2simple_rnn_cell_173_matmul_readvariableop_resource:PdA
3simple_rnn_cell_173_biasadd_readvariableop_resource:dF
4simple_rnn_cell_173_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_173/BiasAdd/ReadVariableOp?)simple_rnn_cell_173/MatMul/ReadVariableOp?+simple_rnn_cell_173/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_173_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_173/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_173_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_173/BiasAddBiasAdd$simple_rnn_cell_173/MatMul:product:02simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_173_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_173/MatMul_1MatMulzeros:output:03simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_173/addAddV2$simple_rnn_cell_173/BiasAdd:output:0&simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_173/TanhTanhsimple_rnn_cell_173/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_173_matmul_readvariableop_resource3simple_rnn_cell_173_biasadd_readvariableop_resource4simple_rnn_cell_173_matmul_1_readvariableop_resource*
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
while_body_11269872*
condR
while_cond_11269871*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_173/BiasAdd/ReadVariableOp*^simple_rnn_cell_173/MatMul/ReadVariableOp,^simple_rnn_cell_173/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_173/BiasAdd/ReadVariableOp*simple_rnn_cell_173/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_173/MatMul/ReadVariableOp)simple_rnn_cell_173/MatMul/ReadVariableOp2Z
+simple_rnn_cell_173/MatMul_1/ReadVariableOp+simple_rnn_cell_173/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?-
?
while_body_11269656
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_173_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_173_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_173_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_173_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_173/MatMul/ReadVariableOp?1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_173_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_173/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_173/BiasAddBiasAdd*while/simple_rnn_cell_173/MatMul:product:08while/simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_173/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_173/addAddV2*while/simple_rnn_cell_173/BiasAdd:output:0,while/simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_173/TanhTanh!while/simple_rnn_cell_173/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_173/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_173/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_173/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_173/MatMul/ReadVariableOp2^while/simple_rnn_cell_173/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_173_biasadd_readvariableop_resource;while_simple_rnn_cell_173_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_173_matmul_1_readvariableop_resource<while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_173_matmul_readvariableop_resource:while_simple_rnn_cell_173_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp0while/simple_rnn_cell_173/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_173/MatMul/ReadVariableOp/while/simple_rnn_cell_173/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp1while/simple_rnn_cell_173/MatMul_1/ReadVariableOp: 
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
g
H__inference_dropout_13_layer_call_and_return_conditional_losses_11268118

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
while_body_11269369
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_172_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_172_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_172_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_172_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_172/MatMul/ReadVariableOp?1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_172_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_172/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_172/BiasAddBiasAdd*while/simple_rnn_cell_172/MatMul:product:08while/simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_172/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_172/addAddV2*while/simple_rnn_cell_172/BiasAdd:output:0,while/simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_172/TanhTanh!while/simple_rnn_cell_172/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_172/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_172/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_172/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_172/MatMul/ReadVariableOp2^while/simple_rnn_cell_172/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_172_biasadd_readvariableop_resource;while_simple_rnn_cell_172_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_172_matmul_1_readvariableop_resource<while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_172_matmul_readvariableop_resource:while_simple_rnn_cell_172_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp0while/simple_rnn_cell_172/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_172/MatMul/ReadVariableOp/while/simple_rnn_cell_172/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp1while/simple_rnn_cell_172/MatMul_1/ReadVariableOp: 
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
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11267557

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

?
6__inference_simple_rnn_cell_172_layer_call_fn_11270120

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
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11267385o
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
?
f
H__inference_dropout_13_layer_call_and_return_conditional_losses_11270061

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
?
?
while_cond_11267569
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11267569___redundant_placeholder06
2while_while_cond_11267569___redundant_placeholder16
2while_while_cond_11267569___redundant_placeholder26
2while_while_cond_11267569___redundant_placeholder3
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
?:
?
!simple_rnn_12_while_body_112688528
4simple_rnn_12_while_simple_rnn_12_while_loop_counter>
:simple_rnn_12_while_simple_rnn_12_while_maximum_iterations#
simple_rnn_12_while_placeholder%
!simple_rnn_12_while_placeholder_1%
!simple_rnn_12_while_placeholder_27
3simple_rnn_12_while_simple_rnn_12_strided_slice_1_0s
osimple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_12_tensorarrayunstack_tensorlistfromtensor_0Z
Hsimple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resource_0:PW
Isimple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resource_0:P\
Jsimple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0:PP 
simple_rnn_12_while_identity"
simple_rnn_12_while_identity_1"
simple_rnn_12_while_identity_2"
simple_rnn_12_while_identity_3"
simple_rnn_12_while_identity_45
1simple_rnn_12_while_simple_rnn_12_strided_slice_1q
msimple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_12_tensorarrayunstack_tensorlistfromtensorX
Fsimple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resource:PU
Gsimple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resource:PZ
Hsimple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resource:PP??>simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOp?=simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOp??simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp?
Esimple_rnn_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
7simple_rnn_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_12_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_12_while_placeholderNsimple_rnn_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
=simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOpHsimple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
.simple_rnn_12/while/simple_rnn_cell_172/MatMulMatMul>simple_rnn_12/while/TensorArrayV2Read/TensorListGetItem:item:0Esimple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
>simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOpIsimple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
/simple_rnn_12/while/simple_rnn_cell_172/BiasAddBiasAdd8simple_rnn_12/while/simple_rnn_cell_172/MatMul:product:0Fsimple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
?simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOpJsimple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
0simple_rnn_12/while/simple_rnn_cell_172/MatMul_1MatMul!simple_rnn_12_while_placeholder_2Gsimple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_12/while/simple_rnn_cell_172/addAddV28simple_rnn_12/while/simple_rnn_cell_172/BiasAdd:output:0:simple_rnn_12/while/simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
,simple_rnn_12/while/simple_rnn_cell_172/TanhTanh/simple_rnn_12/while/simple_rnn_cell_172/add:z:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_12_while_placeholder_1simple_rnn_12_while_placeholder0simple_rnn_12/while/simple_rnn_cell_172/Tanh:y:0*
_output_shapes
: *
element_dtype0:???[
simple_rnn_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_12/while/addAddV2simple_rnn_12_while_placeholder"simple_rnn_12/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_12/while/add_1AddV24simple_rnn_12_while_simple_rnn_12_while_loop_counter$simple_rnn_12/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_12/while/IdentityIdentitysimple_rnn_12/while/add_1:z:0^simple_rnn_12/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_12/while/Identity_1Identity:simple_rnn_12_while_simple_rnn_12_while_maximum_iterations^simple_rnn_12/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_12/while/Identity_2Identitysimple_rnn_12/while/add:z:0^simple_rnn_12/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_12/while/Identity_3IdentityHsimple_rnn_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_12/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_12/while/Identity_4Identity0simple_rnn_12/while/simple_rnn_cell_172/Tanh:y:0^simple_rnn_12/while/NoOp*
T0*'
_output_shapes
:?????????P?
simple_rnn_12/while/NoOpNoOp?^simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOp>^simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOp@^simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_12_while_identity%simple_rnn_12/while/Identity:output:0"I
simple_rnn_12_while_identity_1'simple_rnn_12/while/Identity_1:output:0"I
simple_rnn_12_while_identity_2'simple_rnn_12/while/Identity_2:output:0"I
simple_rnn_12_while_identity_3'simple_rnn_12/while/Identity_3:output:0"I
simple_rnn_12_while_identity_4'simple_rnn_12/while/Identity_4:output:0"h
1simple_rnn_12_while_simple_rnn_12_strided_slice_13simple_rnn_12_while_simple_rnn_12_strided_slice_1_0"?
Gsimple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resourceIsimple_rnn_12_while_simple_rnn_cell_172_biasadd_readvariableop_resource_0"?
Hsimple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resourceJsimple_rnn_12_while_simple_rnn_cell_172_matmul_1_readvariableop_resource_0"?
Fsimple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resourceHsimple_rnn_12_while_simple_rnn_cell_172_matmul_readvariableop_resource_0"?
msimple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_12_tensorarrayunstack_tensorlistfromtensorosimple_rnn_12_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2?
>simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOp>simple_rnn_12/while/simple_rnn_cell_172/BiasAdd/ReadVariableOp2~
=simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOp=simple_rnn_12/while/simple_rnn_cell_172/MatMul/ReadVariableOp2?
?simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp?simple_rnn_12/while/simple_rnn_cell_172/MatMul_1/ReadVariableOp: 
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
?
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268069

inputs(
simple_rnn_12_11267916:P$
simple_rnn_12_11267918:P(
simple_rnn_12_11267920:PP(
simple_rnn_13_11268038:Pd$
simple_rnn_13_11268040:d(
simple_rnn_13_11268042:dd"
dense_6_11268063:d
dense_6_11268065:
identity??dense_6/StatefulPartitionedCall?%simple_rnn_12/StatefulPartitionedCall?%simple_rnn_13/StatefulPartitionedCall?
%simple_rnn_12/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_12_11267916simple_rnn_12_11267918simple_rnn_12_11267920*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11267915?
dropout_12/PartitionedCallPartitionedCall.simple_rnn_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_11267928?
%simple_rnn_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0simple_rnn_13_11268038simple_rnn_13_11268040simple_rnn_13_11268042*
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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11268037?
dropout_13/PartitionedCallPartitionedCall.simple_rnn_13/StatefulPartitionedCall:output:0*
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
H__inference_dropout_13_layer_call_and_return_conditional_losses_11268050?
dense_6/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_6_11268063dense_6_11268065*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_11268062w
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_6/StatefulPartitionedCall&^simple_rnn_12/StatefulPartitionedCall&^simple_rnn_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2N
%simple_rnn_12/StatefulPartitionedCall%simple_rnn_12/StatefulPartitionedCall2N
%simple_rnn_13/StatefulPartitionedCall%simple_rnn_13/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
!simple_rnn_13_while_cond_112689638
4simple_rnn_13_while_simple_rnn_13_while_loop_counter>
:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations#
simple_rnn_13_while_placeholder%
!simple_rnn_13_while_placeholder_1%
!simple_rnn_13_while_placeholder_2:
6simple_rnn_13_while_less_simple_rnn_13_strided_slice_1R
Nsimple_rnn_13_while_simple_rnn_13_while_cond_11268963___redundant_placeholder0R
Nsimple_rnn_13_while_simple_rnn_13_while_cond_11268963___redundant_placeholder1R
Nsimple_rnn_13_while_simple_rnn_13_while_cond_11268963___redundant_placeholder2R
Nsimple_rnn_13_while_simple_rnn_13_while_cond_11268963___redundant_placeholder3 
simple_rnn_13_while_identity
?
simple_rnn_13/while/LessLesssimple_rnn_13_while_placeholder6simple_rnn_13_while_less_simple_rnn_13_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_13/while/IdentityIdentitysimple_rnn_13/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_13_while_identity%simple_rnn_13/while/Identity:output:0*(
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
?
I
-__inference_dropout_13_layer_call_fn_11270051

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
H__inference_dropout_13_layer_call_and_return_conditional_losses_11268050`
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
?
?
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11267677

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
?
/__inference_sequential_6_layer_call_fn_11268590

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
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268452o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?=
?
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11268242

inputsD
2simple_rnn_cell_173_matmul_readvariableop_resource:PdA
3simple_rnn_cell_173_biasadd_readvariableop_resource:dF
4simple_rnn_cell_173_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_173/BiasAdd/ReadVariableOp?)simple_rnn_cell_173/MatMul/ReadVariableOp?+simple_rnn_cell_173/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_173_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_173/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_173_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_173/BiasAddBiasAdd$simple_rnn_cell_173/MatMul:product:02simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_173_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_173/MatMul_1MatMulzeros:output:03simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_173/addAddV2$simple_rnn_cell_173/BiasAdd:output:0&simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_173/TanhTanhsimple_rnn_cell_173/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_173_matmul_readvariableop_resource3simple_rnn_cell_173_biasadd_readvariableop_resource4simple_rnn_cell_173_matmul_1_readvariableop_resource*
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
while_body_11268176*
condR
while_cond_11268175*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_173/BiasAdd/ReadVariableOp*^simple_rnn_cell_173/MatMul/ReadVariableOp,^simple_rnn_cell_173/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_173/BiasAdd/ReadVariableOp*simple_rnn_cell_173/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_173/MatMul/ReadVariableOp)simple_rnn_cell_173/MatMul/ReadVariableOp2Z
+simple_rnn_cell_173/MatMul_1/ReadVariableOp+simple_rnn_cell_173/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11267385

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
?
?
while_cond_11269152
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11269152___redundant_placeholder06
2while_while_cond_11269152___redundant_placeholder16
2while_while_cond_11269152___redundant_placeholder26
2while_while_cond_11269152___redundant_placeholder3
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
while_body_11267437
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_172_11267459_0:P2
$while_simple_rnn_cell_172_11267461_0:P6
$while_simple_rnn_cell_172_11267463_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_172_11267459:P0
"while_simple_rnn_cell_172_11267461:P4
"while_simple_rnn_cell_172_11267463:PP??1while/simple_rnn_cell_172/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/simple_rnn_cell_172/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_172_11267459_0$while_simple_rnn_cell_172_11267461_0$while_simple_rnn_cell_172_11267463_0*
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
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11267385?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_172/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_172/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp2^while/simple_rnn_cell_172/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_172_11267459$while_simple_rnn_cell_172_11267459_0"J
"while_simple_rnn_cell_172_11267461$while_simple_rnn_cell_172_11267461_0"J
"while_simple_rnn_cell_172_11267463$while_simple_rnn_cell_172_11267463_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2f
1while/simple_rnn_cell_172/StatefulPartitionedCall1while/simple_rnn_cell_172/StatefulPartitionedCall: 
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
?
.sequential_6_simple_rnn_12_while_cond_11267038R
Nsequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_while_loop_counterX
Tsequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_while_maximum_iterations0
,sequential_6_simple_rnn_12_while_placeholder2
.sequential_6_simple_rnn_12_while_placeholder_12
.sequential_6_simple_rnn_12_while_placeholder_2T
Psequential_6_simple_rnn_12_while_less_sequential_6_simple_rnn_12_strided_slice_1l
hsequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_while_cond_11267038___redundant_placeholder0l
hsequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_while_cond_11267038___redundant_placeholder1l
hsequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_while_cond_11267038___redundant_placeholder2l
hsequential_6_simple_rnn_12_while_sequential_6_simple_rnn_12_while_cond_11267038___redundant_placeholder3-
)sequential_6_simple_rnn_12_while_identity
?
%sequential_6/simple_rnn_12/while/LessLess,sequential_6_simple_rnn_12_while_placeholderPsequential_6_simple_rnn_12_while_less_sequential_6_simple_rnn_12_strided_slice_1*
T0*
_output_shapes
: ?
)sequential_6/simple_rnn_12/while/IdentityIdentity)sequential_6/simple_rnn_12/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_6_simple_rnn_12_while_identity2sequential_6/simple_rnn_12/while/Identity:output:0*(
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
?
f
-__inference_dropout_13_layer_call_fn_11270056

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
H__inference_dropout_13_layer_call_and_return_conditional_losses_11268118o
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
?=
?
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11269543

inputsD
2simple_rnn_cell_172_matmul_readvariableop_resource:PA
3simple_rnn_cell_172_biasadd_readvariableop_resource:PF
4simple_rnn_cell_172_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_172/BiasAdd/ReadVariableOp?)simple_rnn_cell_172/MatMul/ReadVariableOp?+simple_rnn_cell_172/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_172_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_172/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_172_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_172/BiasAddBiasAdd$simple_rnn_cell_172/MatMul:product:02simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_172_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_172/MatMul_1MatMulzeros:output:03simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_172/addAddV2$simple_rnn_cell_172/BiasAdd:output:0&simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_172/TanhTanhsimple_rnn_cell_172/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_172_matmul_readvariableop_resource3simple_rnn_cell_172_biasadd_readvariableop_resource4simple_rnn_cell_172_matmul_1_readvariableop_resource*
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
while_body_11269477*
condR
while_cond_11269476*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_172/BiasAdd/ReadVariableOp*^simple_rnn_cell_172/MatMul/ReadVariableOp,^simple_rnn_cell_172/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_172/BiasAdd/ReadVariableOp*simple_rnn_cell_172/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_172/MatMul/ReadVariableOp)simple_rnn_cell_172/MatMul/ReadVariableOp2Z
+simple_rnn_cell_172/MatMul_1/ReadVariableOp+simple_rnn_cell_172/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
H__inference_dropout_12_layer_call_and_return_conditional_losses_11269558

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?!
?
while_body_11267570
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_173_11267592_0:Pd2
$while_simple_rnn_cell_173_11267594_0:d6
$while_simple_rnn_cell_173_11267596_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_173_11267592:Pd0
"while_simple_rnn_cell_173_11267594:d4
"while_simple_rnn_cell_173_11267596:dd??1while/simple_rnn_cell_173/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
1while/simple_rnn_cell_173/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_173_11267592_0$while_simple_rnn_cell_173_11267594_0$while_simple_rnn_cell_173_11267596_0*
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
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11267557?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_173/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_173/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp2^while/simple_rnn_cell_173/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_173_11267592$while_simple_rnn_cell_173_11267592_0"J
"while_simple_rnn_cell_173_11267594$while_simple_rnn_cell_173_11267594_0"J
"while_simple_rnn_cell_173_11267596$while_simple_rnn_cell_173_11267596_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2f
1while/simple_rnn_cell_173/StatefulPartitionedCall1while/simple_rnn_cell_173/StatefulPartitionedCall: 
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
while_cond_11267848
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11267848___redundant_placeholder06
2while_while_cond_11267848___redundant_placeholder16
2while_while_cond_11267848___redundant_placeholder26
2while_while_cond_11267848___redundant_placeholder3
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
/__inference_sequential_6_layer_call_fn_11268569

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
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268069o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?>
?
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11269219
inputs_0D
2simple_rnn_cell_172_matmul_readvariableop_resource:PA
3simple_rnn_cell_172_biasadd_readvariableop_resource:PF
4simple_rnn_cell_172_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_172/BiasAdd/ReadVariableOp?)simple_rnn_cell_172/MatMul/ReadVariableOp?+simple_rnn_cell_172/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_172/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_172_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_172/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_172/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_172/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_172_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_172/BiasAddBiasAdd$simple_rnn_cell_172/MatMul:product:02simple_rnn_cell_172/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_172/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_172_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_172/MatMul_1MatMulzeros:output:03simple_rnn_cell_172/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_172/addAddV2$simple_rnn_cell_172/BiasAdd:output:0&simple_rnn_cell_172/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_172/TanhTanhsimple_rnn_cell_172/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_172_matmul_readvariableop_resource3simple_rnn_cell_172_biasadd_readvariableop_resource4simple_rnn_cell_172_matmul_1_readvariableop_resource*
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
while_body_11269153*
condR
while_cond_11269152*8
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
NoOpNoOp+^simple_rnn_cell_172/BiasAdd/ReadVariableOp*^simple_rnn_cell_172/MatMul/ReadVariableOp,^simple_rnn_cell_172/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2X
*simple_rnn_cell_172/BiasAdd/ReadVariableOp*simple_rnn_cell_172/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_172/MatMul/ReadVariableOp)simple_rnn_cell_172/MatMul/ReadVariableOp2Z
+simple_rnn_cell_172/MatMul_1/ReadVariableOp+simple_rnn_cell_172/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_11267728
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11267728___redundant_placeholder06
2while_while_cond_11267728___redundant_placeholder16
2while_while_cond_11267728___redundant_placeholder26
2while_while_cond_11267728___redundant_placeholder3
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
?
?
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11267265

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
?:
?
!simple_rnn_13_while_body_112687378
4simple_rnn_13_while_simple_rnn_13_while_loop_counter>
:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations#
simple_rnn_13_while_placeholder%
!simple_rnn_13_while_placeholder_1%
!simple_rnn_13_while_placeholder_27
3simple_rnn_13_while_simple_rnn_13_strided_slice_1_0s
osimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0Z
Hsimple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resource_0:PdW
Isimple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resource_0:d\
Jsimple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0:dd 
simple_rnn_13_while_identity"
simple_rnn_13_while_identity_1"
simple_rnn_13_while_identity_2"
simple_rnn_13_while_identity_3"
simple_rnn_13_while_identity_45
1simple_rnn_13_while_simple_rnn_13_strided_slice_1q
msimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorX
Fsimple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resource:PdU
Gsimple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resource:dZ
Hsimple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resource:dd??>simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOp?=simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOp??simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp?
Esimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
7simple_rnn_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_13_while_placeholderNsimple_rnn_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
=simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOpReadVariableOpHsimple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
.simple_rnn_13/while/simple_rnn_cell_173/MatMulMatMul>simple_rnn_13/while/TensorArrayV2Read/TensorListGetItem:item:0Esimple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
>simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOpReadVariableOpIsimple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
/simple_rnn_13/while/simple_rnn_cell_173/BiasAddBiasAdd8simple_rnn_13/while/simple_rnn_cell_173/MatMul:product:0Fsimple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
?simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOpReadVariableOpJsimple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
0simple_rnn_13/while/simple_rnn_cell_173/MatMul_1MatMul!simple_rnn_13_while_placeholder_2Gsimple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_13/while/simple_rnn_cell_173/addAddV28simple_rnn_13/while/simple_rnn_cell_173/BiasAdd:output:0:simple_rnn_13/while/simple_rnn_cell_173/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
,simple_rnn_13/while/simple_rnn_cell_173/TanhTanh/simple_rnn_13/while/simple_rnn_cell_173/add:z:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_13_while_placeholder_1simple_rnn_13_while_placeholder0simple_rnn_13/while/simple_rnn_cell_173/Tanh:y:0*
_output_shapes
: *
element_dtype0:???[
simple_rnn_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_13/while/addAddV2simple_rnn_13_while_placeholder"simple_rnn_13/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_13/while/add_1AddV24simple_rnn_13_while_simple_rnn_13_while_loop_counter$simple_rnn_13/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_13/while/IdentityIdentitysimple_rnn_13/while/add_1:z:0^simple_rnn_13/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_13/while/Identity_1Identity:simple_rnn_13_while_simple_rnn_13_while_maximum_iterations^simple_rnn_13/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_13/while/Identity_2Identitysimple_rnn_13/while/add:z:0^simple_rnn_13/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_13/while/Identity_3IdentityHsimple_rnn_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_13/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_13/while/Identity_4Identity0simple_rnn_13/while/simple_rnn_cell_173/Tanh:y:0^simple_rnn_13/while/NoOp*
T0*'
_output_shapes
:?????????d?
simple_rnn_13/while/NoOpNoOp?^simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOp>^simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOp@^simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_13_while_identity%simple_rnn_13/while/Identity:output:0"I
simple_rnn_13_while_identity_1'simple_rnn_13/while/Identity_1:output:0"I
simple_rnn_13_while_identity_2'simple_rnn_13/while/Identity_2:output:0"I
simple_rnn_13_while_identity_3'simple_rnn_13/while/Identity_3:output:0"I
simple_rnn_13_while_identity_4'simple_rnn_13/while/Identity_4:output:0"h
1simple_rnn_13_while_simple_rnn_13_strided_slice_13simple_rnn_13_while_simple_rnn_13_strided_slice_1_0"?
Gsimple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resourceIsimple_rnn_13_while_simple_rnn_cell_173_biasadd_readvariableop_resource_0"?
Hsimple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resourceJsimple_rnn_13_while_simple_rnn_cell_173_matmul_1_readvariableop_resource_0"?
Fsimple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resourceHsimple_rnn_13_while_simple_rnn_cell_173_matmul_readvariableop_resource_0"?
msimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensorosimple_rnn_13_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2?
>simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOp>simple_rnn_13/while/simple_rnn_cell_173/BiasAdd/ReadVariableOp2~
=simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOp=simple_rnn_13/while/simple_rnn_cell_173/MatMul/ReadVariableOp2?
?simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp?simple_rnn_13/while/simple_rnn_cell_173/MatMul_1/ReadVariableOp: 
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
W
simple_rnn_12_input@
%serving_default_simple_rnn_12_input:0?????????;
dense_60
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
/__inference_sequential_6_layer_call_fn_11268088
/__inference_sequential_6_layer_call_fn_11268569
/__inference_sequential_6_layer_call_fn_11268590
/__inference_sequential_6_layer_call_fn_11268492?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268810
J__inference_sequential_6_layer_call_and_return_conditional_losses_11269044
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268517
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268542?
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
#__inference__wrapped_model_11267217simple_rnn_12_input"?
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
0__inference_simple_rnn_12_layer_call_fn_11269078
0__inference_simple_rnn_12_layer_call_fn_11269089
0__inference_simple_rnn_12_layer_call_fn_11269100
0__inference_simple_rnn_12_layer_call_fn_11269111?
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
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11269219
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11269327
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11269435
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11269543?
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
-__inference_dropout_12_layer_call_fn_11269548
-__inference_dropout_12_layer_call_fn_11269553?
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
H__inference_dropout_12_layer_call_and_return_conditional_losses_11269558
H__inference_dropout_12_layer_call_and_return_conditional_losses_11269570?
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
0__inference_simple_rnn_13_layer_call_fn_11269581
0__inference_simple_rnn_13_layer_call_fn_11269592
0__inference_simple_rnn_13_layer_call_fn_11269603
0__inference_simple_rnn_13_layer_call_fn_11269614?
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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11269722
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11269830
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11269938
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11270046?
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
-__inference_dropout_13_layer_call_fn_11270051
-__inference_dropout_13_layer_call_fn_11270056?
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
H__inference_dropout_13_layer_call_and_return_conditional_losses_11270061
H__inference_dropout_13_layer_call_and_return_conditional_losses_11270073?
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
 :d2dense_6/kernel
:2dense_6/bias
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
*__inference_dense_6_layer_call_fn_11270082?
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
E__inference_dense_6_layer_call_and_return_conditional_losses_11270092?
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
&__inference_signature_wrapper_11269067simple_rnn_12_input"?
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
9:7P2'simple_rnn_12/simple_rnn_cell_12/kernel
C:APP21simple_rnn_12/simple_rnn_cell_12/recurrent_kernel
3:1P2%simple_rnn_12/simple_rnn_cell_12/bias
9:7Pd2'simple_rnn_13/simple_rnn_cell_13/kernel
C:Add21simple_rnn_13/simple_rnn_cell_13/recurrent_kernel
3:1d2%simple_rnn_13/simple_rnn_cell_13/bias
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
6__inference_simple_rnn_cell_172_layer_call_fn_11270106
6__inference_simple_rnn_cell_172_layer_call_fn_11270120?
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
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11270137
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11270154?
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
6__inference_simple_rnn_cell_173_layer_call_fn_11270168
6__inference_simple_rnn_cell_173_layer_call_fn_11270182?
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
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11270199
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11270216?
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
%:#d2Adam/dense_6/kernel/m
:2Adam/dense_6/bias/m
>:<P2.Adam/simple_rnn_12/simple_rnn_cell_12/kernel/m
H:FPP28Adam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/m
8:6P2,Adam/simple_rnn_12/simple_rnn_cell_12/bias/m
>:<Pd2.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/m
H:Fdd28Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/m
8:6d2,Adam/simple_rnn_13/simple_rnn_cell_13/bias/m
%:#d2Adam/dense_6/kernel/v
:2Adam/dense_6/bias/v
>:<P2.Adam/simple_rnn_12/simple_rnn_cell_12/kernel/v
H:FPP28Adam/simple_rnn_12/simple_rnn_cell_12/recurrent_kernel/v
8:6P2,Adam/simple_rnn_12/simple_rnn_cell_12/bias/v
>:<Pd2.Adam/simple_rnn_13/simple_rnn_cell_13/kernel/v
H:Fdd28Adam/simple_rnn_13/simple_rnn_cell_13/recurrent_kernel/v
8:6d2,Adam/simple_rnn_13/simple_rnn_cell_13/bias/v?
#__inference__wrapped_model_11267217ACBDFE23@?=
6?3
1?.
simple_rnn_12_input?????????
? "1?.
,
dense_6!?
dense_6??????????
E__inference_dense_6_layer_call_and_return_conditional_losses_11270092\23/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? }
*__inference_dense_6_layer_call_fn_11270082O23/?,
%?"
 ?
inputs?????????d
? "???????????
H__inference_dropout_12_layer_call_and_return_conditional_losses_11269558d7?4
-?*
$?!
inputs?????????P
p 
? ")?&
?
0?????????P
? ?
H__inference_dropout_12_layer_call_and_return_conditional_losses_11269570d7?4
-?*
$?!
inputs?????????P
p
? ")?&
?
0?????????P
? ?
-__inference_dropout_12_layer_call_fn_11269548W7?4
-?*
$?!
inputs?????????P
p 
? "??????????P?
-__inference_dropout_12_layer_call_fn_11269553W7?4
-?*
$?!
inputs?????????P
p
? "??????????P?
H__inference_dropout_13_layer_call_and_return_conditional_losses_11270061\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? ?
H__inference_dropout_13_layer_call_and_return_conditional_losses_11270073\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
-__inference_dropout_13_layer_call_fn_11270051O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
-__inference_dropout_13_layer_call_fn_11270056O3?0
)?&
 ?
inputs?????????d
p
? "??????????d?
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268517{ACBDFE23H?E
>?;
1?.
simple_rnn_12_input?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268542{ACBDFE23H?E
>?;
1?.
simple_rnn_12_input?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_11268810nACBDFE23;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_11269044nACBDFE23;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_6_layer_call_fn_11268088nACBDFE23H?E
>?;
1?.
simple_rnn_12_input?????????
p 

 
? "???????????
/__inference_sequential_6_layer_call_fn_11268492nACBDFE23H?E
>?;
1?.
simple_rnn_12_input?????????
p

 
? "???????????
/__inference_sequential_6_layer_call_fn_11268569aACBDFE23;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
/__inference_sequential_6_layer_call_fn_11268590aACBDFE23;?8
1?.
$?!
inputs?????????
p

 
? "???????????
&__inference_signature_wrapper_11269067?ACBDFE23W?T
? 
M?J
H
simple_rnn_12_input1?.
simple_rnn_12_input?????????"1?.
,
dense_6!?
dense_6??????????
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11269219?ACBO?L
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
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11269327?ACBO?L
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
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11269435qACB??<
5?2
$?!
inputs?????????

 
p 

 
? ")?&
?
0?????????P
? ?
K__inference_simple_rnn_12_layer_call_and_return_conditional_losses_11269543qACB??<
5?2
$?!
inputs?????????

 
p

 
? ")?&
?
0?????????P
? ?
0__inference_simple_rnn_12_layer_call_fn_11269078}ACBO?L
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
0__inference_simple_rnn_12_layer_call_fn_11269089}ACBO?L
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
0__inference_simple_rnn_12_layer_call_fn_11269100dACB??<
5?2
$?!
inputs?????????

 
p 

 
? "??????????P?
0__inference_simple_rnn_12_layer_call_fn_11269111dACB??<
5?2
$?!
inputs?????????

 
p

 
? "??????????P?
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11269722}DFEO?L
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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11269830}DFEO?L
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
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11269938mDFE??<
5?2
$?!
inputs?????????P

 
p 

 
? "%?"
?
0?????????d
? ?
K__inference_simple_rnn_13_layer_call_and_return_conditional_losses_11270046mDFE??<
5?2
$?!
inputs?????????P

 
p

 
? "%?"
?
0?????????d
? ?
0__inference_simple_rnn_13_layer_call_fn_11269581pDFEO?L
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
0__inference_simple_rnn_13_layer_call_fn_11269592pDFEO?L
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
0__inference_simple_rnn_13_layer_call_fn_11269603`DFE??<
5?2
$?!
inputs?????????P

 
p 

 
? "??????????d?
0__inference_simple_rnn_13_layer_call_fn_11269614`DFE??<
5?2
$?!
inputs?????????P

 
p

 
? "??????????d?
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11270137?ACB\?Y
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
Q__inference_simple_rnn_cell_172_layer_call_and_return_conditional_losses_11270154?ACB\?Y
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
6__inference_simple_rnn_cell_172_layer_call_fn_11270106?ACB\?Y
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
6__inference_simple_rnn_cell_172_layer_call_fn_11270120?ACB\?Y
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
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11270199?DFE\?Y
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
Q__inference_simple_rnn_cell_173_layer_call_and_return_conditional_losses_11270216?DFE\?Y
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
6__inference_simple_rnn_cell_173_layer_call_fn_11270168?DFE\?Y
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
6__inference_simple_rnn_cell_173_layer_call_fn_11270182?DFE\?Y
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