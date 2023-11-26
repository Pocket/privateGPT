import {Construct} from "constructs";
import {SagemakerEndpointConfiguration} from "@cdktf/provider-aws/lib/sagemaker-endpoint-configuration";
import { config } from './config';
import {SagemakerEndpoint} from "@cdktf/provider-aws/lib/sagemaker-endpoint";
import {SagemakerModel} from "@cdktf/provider-aws/lib/sagemaker-model";
import {DataAwsIamPolicyDocument} from "@cdktf/provider-aws/lib/data-aws-iam-policy-document";
import {IamRole} from "@cdktf/provider-aws/lib/iam-role";
import {ApplicationECR, PocketVPC} from "@pocket-tools/terraform-modules";
import {S3Bucket} from "@cdktf/provider-aws/lib/s3-bucket";
import {IamRolePolicy} from "@cdktf/provider-aws/lib/iam-role-policy";
import {Resource} from "@cdktf/provider-null/lib/resource";
import {S3Object} from "@cdktf/provider-aws/lib/s3-object";
import {Fn, ITerraformDependable, Token} from "cdktf";
import {DataLocalFile} from "@cdktf/provider-local/lib/data-local-file";

export class Sagemaker extends Construct {
  public readonly llmEndpoint: ApplicationSagemaker;
  public readonly embeddingsEndpoint: ApplicationSagemaker;
  public readonly modelStorage: S3Bucket;


  constructor(scope: Construct, name: string, vpc: PocketVPC) {
        super(scope, name);
        this.modelStorage = this.createModelStorage();

        this.llmEndpoint = new ApplicationSagemaker(this, `llm`, {
            // The model we are using requires a GPU ml.g4dn.xlarge is the cheapest instance with a gpu, but we use ml.g5.2xlarge because this model requires more memory allocation.
            instanceType: 'ml.g5.2xlarge',
            instanceCount: 1,
            vpc,
            modelName: 'llm',
            environment: {
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": vpc.region,
                'SM_NUM_GPUS': '1', // Number of GPU used per replica

                // The sagemaker setup in the llm code expects a specific format, which is what the text-generation pipeline enables.
                // You can look at the model on HuggingFace to determine what the HF_TASK needs to be set to.
                // If not set, the image will try and determine it from the Architecture set in Config.json of your model
                // https://github.com/aws/sagemaker-huggingface-inference-toolkit/blob/main/src/sagemaker_huggingface_inference_toolkit/transformers_utils.py#L80-L92
                'HF_TASK': 'text-generation'
            },
           modelStorage: this.modelStorage,
           model: this.uploadLLMModel(this.modelStorage)
        });

       this.embeddingsEndpoint = new ApplicationSagemaker(this, `embeddings`, {
            // The model we are using requires a GPU ml.g4dn.xlarge is the cheapest instance with a gpu.
            instanceType: 'ml.g4dn.xlarge',
            instanceCount: 1,
            vpc,
            modelName: 'embeddings',
            environment: {
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": vpc.region,
                'SM_NUM_GPUS': '1', // Number of GPU used per replica
                'HF_TASK': 'text-classification'
            },
           modelStorage: this.modelStorage,
           model: this.uploadEmbeddingModel(this.modelStorage)
       });
   }

   private uploadEmbeddingModel(modelStorage: S3Bucket): S3Object {
        const buildModel = new Resource(this, 'build-embedding', {
          triggers: {
              // Sets this null resource to be triggered on every terraform apply
              alwaysRun: Fn.timestamp(),
          },
          dependsOn: [
            modelStorage,
          ],
        });


        const inferenceCode = `
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as f

##
# The hugging face no code deployment of Sagemaker Inference at
# https://github.com/aws/sagemaker-huggingface-inference-toolkit
# Does not support BertTokenizer models out of the box.
# So for embedding models we push our own to AWS and add this hook script that is supported at
# https://github.com/aws/sagemaker-huggingface-inference-toolkit#-user-defined-codemodules
# This lets us use Bert based models and return data in the format expected by privateGPTs embedding
# Example: https://github.com/huggingface/notebooks/blob/main/sagemaker/17_custom_inference_script/sagemaker-notebook.ipynb
##

# Helper: Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def model_fn(model_dir):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    return model, tokenizer


def predict_fn(data, model_and_tokenizer):
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer

    # Tokenize sentences
    sentences = data.pop('inputs', data)
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = f.normalize(sentence_embeddings, p=2, dim=1)

    # return dictionary, which will be json serializable
    return {'vectors': sentence_embeddings.tolist()}
        `

        const modelScript = `
pip install -U "huggingface_hub[cli]"
rm -rf ./embedding-model
huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir=./embedding-model --local-dir-use-symlinks=False
mkdir ./embedding-model/code
echo "${inferenceCode}" > ./embedding-model/code/inference.py
tar -C embedding-model -zcvf embedding-model.tar.gz .
`

        buildModel.addOverride(
          'provisioner.local-exec.command',
          modelScript,
        );

        return new S3Object(this, 'embedding_model_object', {
            bucket: modelStorage.bucket,
            key: `embeddings/model-${Fn.uuid()}.tar.gz`,
            source: './embedding-model.tar.gz',
            dependsOn: [buildModel]
        });
   }

   private uploadLLMModel(modelStorage: S3Bucket): S3Object {
        const buildModel = new Resource(this, 'build-llm', {
          triggers: {
              // Sets this null resource to be triggered on every terraform apply
              alwaysRun: Fn.timestamp(),
          },
          dependsOn: [
            modelStorage,
          ],
        });

        const inferenceCode = `
from transformers import AutoTokenizer, AutoModel
import pip

##
# The hugging face no code deployment of Sagemaker Inference at
# https://github.com/aws/sagemaker-huggingface-inference-toolkit
# Does not support the latest version of the mistral images,
# So for mistral models we push our own to AWS and add this hook script that is supported at
# https://github.com/aws/sagemaker-huggingface-inference-toolkit#-user-defined-codemodules
# This lets us install the latest version of transformers that should support this.
# Example: https://github.com/huggingface/notebooks/blob/main/sagemaker/17_custom_inference_script/sagemaker-notebook.ipynb
##

def model_fn(model_dir):
    # Current version of the inference image does not have the latest transformers needed to support Mistral based images
    pip.main(['install', 'transformers[sentencepiece,audio,vision]==4.35.2'])
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    return model, tokenizer
`

        const modelScript = `
pip install -U "huggingface_hub[cli]"
rm -rf ./llm-model
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir=./llm-model --local-dir-use-symlinks=False
mkdir ./llm-model/code
echo "${inferenceCode}" > ./llm-model/code/inference.py
tar -C llm-model -zcvf llm-model.tar.gz .
`

        buildModel.addOverride(
          'provisioner.local-exec.command',
          modelScript,
        );

        return new S3Object(this, 'llm_model_object', {
            bucket: modelStorage.bucket,
            key: `llm/model-${Fn.uuid()}.tar.gz`,
            source: './llm-model.tar.gz',
            dependsOn: [buildModel]
        });
   }

   private createModelStorage(): S3Bucket {
      return new S3Bucket(this, `model_storage`, {
          bucketPrefix: `${config.prefix}-ModelStorage`.toLowerCase(),
          tags: config.tags
      })
   }
}

class ApplicationSagemaker extends Construct {

    public readonly configuration: SagemakerEndpointConfiguration
    public readonly model: SagemakerModel
    public readonly endpoint: SagemakerEndpoint

  constructor(scope: Construct, name: string, options: {vpc: PocketVPC, modelName: string, instanceType: string, instanceCount: number, environment: {[key: string]: string;}, modelStorage: S3Bucket, model?: S3Object }) {
    super(scope, name);
    const {  vpc, modelName, instanceType, instanceCount, environment, modelStorage, model } = options;
    this.model = this.createModel({vpc, modelName, environment, modelStorage, model })
    this.configuration = this.createConfiguration({model: this.model, modelName, instanceType, instanceCount})
    this.endpoint = this.createEndpoint({configuration: this.configuration, modelName})
  }


  private createModel(options: { vpc: PocketVPC, modelName: string, environment: {[key: string]: string;}, modelStorage: S3Bucket, model?: S3Object}): SagemakerModel {
        const {vpc, modelName, environment, modelStorage, model } = options;
        const role = new IamRole(this, `model_role`, {
            name: `${config.prefix}-${modelName}-ExecutionRole`,
            assumeRolePolicy: new DataAwsIamPolicyDocument(this, `model_assume_role_policy`, {
                statement: [{
                    actions: ["sts:AssumeRole"],
                    principals: [
                        {
                            type: "Service",
                            identifiers: ["sagemaker.amazonaws.com"]
                        }
                    ]
                }]
                }
            ).json ,
            managedPolicyArns: ['arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'],
            tags: config.tags
        })

      const rolePolicy = new IamRolePolicy(this, `model_role_policy`, {
            name: `${config.prefix}-${modelName}-ExecutionPolicyAdd`,
            policy: new DataAwsIamPolicyDocument(this, `model_role_policy_document`, {
                statement: [{
                    effect: "Allow",
                    actions: [
                        "s3:*"
                    ],
                    resources: [
                      modelStorage.arn,
                      `${modelStorage.arn}/*`
                    ]
                }]
                }
            ).json ,
            role: role.name,
        })

        let depends: ITerraformDependable[] = [role, rolePolicy];
        if (model != undefined) {
            depends = [...depends, model];
        }

        return new SagemakerModel(this, `model`, {
            name: `${config.prefix}-${modelName}`,
            primaryContainer: {
                modelDataUrl : model != undefined ? `s3://${model.bucket}/${model.key}` : undefined,
                // https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-text-generation-inference-containers
                // https://huggingface.co/blog/sagemaker-huggingface-llm
                image: '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04',
                mode: "SingleModel",
                environment
            },
            executionRoleArn: role.arn,
            tags: config.tags,
            // Can't use a vpc until we set up a S3 VPC endpoint
            // vpcConfig: {
            //     subnets: vpc.privateSubnetIds,
            //     securityGroupIds: vpc.internalSecurityGroups.ids
            // },
            dependsOn: depends
        })
    }

     private createConfiguration(options: { model: SagemakerModel, modelName: string, instanceType: string, instanceCount: number }): SagemakerEndpointConfiguration {
      const { model, modelName, instanceType, instanceCount } = options;
      return new SagemakerEndpointConfiguration(this, `config`, {
            name: `${config.prefix}-${modelName}`,
            productionVariants: [{
                variantName: 'variant-1',
                modelName: model.name,
                // Because we are currently using HuggingFace which downloads the model on startup (and needs a startup healthcheck)
                // and because we use a VPC config we can not use Serverless.
                // In the future when we build our own image, we could probably use serverless.
                containerStartupHealthCheckTimeoutInSeconds: 500,
                initialInstanceCount: instanceCount,
                instanceType: instanceType,
            }],
            tags: config.tags,
            dependsOn: [model]
      })
    }

    private createEndpoint(options: {configuration: SagemakerEndpointConfiguration, modelName: string}): SagemakerEndpoint {
        return new SagemakerEndpoint(this, `endpoint`, {
            endpointConfigName: options.configuration.name,
            name: `${config.prefix}-${options.modelName}`,
            deploymentConfig: {
                blueGreenUpdatePolicy: {
                    maximumExecutionTimeoutInSeconds: 600,
                    trafficRoutingConfiguration: {
                        type: "ALL_AT_ONCE",
                        waitIntervalInSeconds: 300
                    }
                }
            },
            tags: config.tags,
            dependsOn: [options.configuration]
        })
    }
}